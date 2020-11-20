import imageio
from pyro.util import torch_isnan
from pyro.infer import Trace_ELBO, TraceEnum_ELBO, TraceMeanField_ELBO
from dataset import *
from torch.utils.data import DataLoader
import torch
from utilities import check_for_checkpt
import numpy as np
from modules import Enc, Dec, RNN_encoder, Combiner, GatedTransition, ID_Layers
from VDSM import VDSM_EncDec, VDSM_Seq
import os
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
import math
import pytz
from datetime import datetime
from scipy.interpolate import interp1d

class Trainer_Tester(object):
    def __init__(self, config):
        # training settings
        self.RUN = config.RUN
        self.bs = config.bs
        self.test_repeats = 1
        self.bs_per_epoch = config.bs_per_epoch
        self.epochs = config.epochs
        self.seed = config.seed
        self.seq_len = config.seq_len
        self.train_VDSMEncDec = config.train_VDSMEncDec
        self.train_VDSMSeq = config.train_VDSMSeq
        self.imsize = 64
        self.nc = 3
        self.model_save_interval = config.model_save_interval
        self.model_test_interval = config.model_test_interval
        self.numsteps = self.bs_per_epoch * self.epochs
        self.dataset_name = config.dataset_name
        self.pretrained_model_VDSMEncDec = config.pretrained_model_VDSMEncDec
        self.pretrained_model_VDSMSeq= config.pretrained_model_VDSMSeq
        self.model_save_path = os.path.join(config.model_save_path, config.RUN)
        self.data_dir = config.data_dir
        self.dataset_dir = os.path.join(self.data_dir, self.dataset_name)
        self.image_path = os.path.join(config.image_path, config.RUN)
        self.num_test_ids = config.num_test_ids
        self.tboard_log = config.tboard_log
        self.log_dir = os.path.join(config.log_dir)
        self.lr_resume = config.lr_resume
        self.test_harness = config.test_harness

        # blended model settings
        self.z_dim = config.z_dim
        self.n_e_w = config.n_e_w
        self.test_temp_id = config.test_temp_id
        self.temp_id_end = config.temp_id_end
        self.temp_id_start = config.temp_id_start
        self.temp_id_frac = config.temp_id_frac
        self.anneal_frac_id = config.anneal_frac_id
        self.anneal_end_id = config.anneal_end_id
        self.anneal_start_id = config.anneal_start_id
        self.anneal_t_midfrac1 = config.anneal_t_midfrac1
        self.anneal_t_midfrac2 = config.anneal_t_midfrac2
        self.anneal_start_t = config.anneal_start_t
        self.anneal_mid_t1 = config.anneal_mid_t1
        self.anneal_mid_t2 = config.anneal_mid_t2
        self.anneal_end_t = config.anneal_end_t
        self.T_0 = self.numsteps // config.T_0_frac
        self.T_mult = config.T_mult
        self.lr_VDSMEncDec = config.lr_VDSMEncDec

        # seq model params
        self.dynamics_dim = config.dynamics_dim
        self.rnn_dropout = config.rnn_dropout
        self.anneal_frac_dynamics = config.anneal_frac_dynamics
        self.anneal_end_dynamics = config.anneal_end_dynamics
        self.anneal_start_dynamics = config.anneal_start_dynamics
        self.trans_dim = config.trans_dim
        self.rnn_dim = config.rnn_dim
        self.rnn_layers = config.rnn_layers
        self.lr_VDSMSeq = config.lr_VDSMSeq
        self.starting_epoch = 0   # this may get overwritten if a checkpoint exists
        self.current_epoch = 0

        if self.train_VDSMSeq:
            assert self.pretrained_model_VDSMEncDec is not None, 'You need to pretrain the VDSM_EncDec first!'

        self.load_data()

        # set up tensorboard logging
        self.tboard = None
        if self.tboard_log:
            config_time_of_run = str(pytz.utc.localize(datetime.utcnow())).split(".")[0][-8:]
            self.tboard = SummaryWriter(
                log_dir=os.path.join(self.log_dir, config.RUN + "_%s/" % config_time_of_run))

        # set up annealing for temperature (blended), anneal_id anneal_t (blended) and anneal_seq (sequential)
        self.temps_id, self.anneals_id, self.anneals_t, self.anneals_dynamics = self.annealing()

        # Initialize models
        self.dev = torch.device('cuda')
        torch.cuda.empty_cache()
        self.enc = Enc(imsize=self.imsize, z_dim=self.z_dim, nc=self.nc, n_expert_components=self.n_e_w).to(self.dev)
        self.dec = Dec(imsize=self.imsize, z_dim=self.z_dim, nc=self.nc, n_expert_components=self.n_e_w).to(self.dev)
        self.id_layers = ID_Layers(n_e_w=self.n_e_w)
        self.VDSM_EncDec = VDSM_EncDec(enc=self.enc, dec=self.dec, id_layers=self.id_layers, seq_len=self.seq_len,
                                       z_dim=self.z_dim, imsize=self.imsize, n_e_w=self.n_e_w,
                                       nc=self.nc).to(self.dev)

        self.rnn_enc = RNN_encoder(input_dim=self.z_dim, rnn_dim=self.rnn_dim, output_dim=self.z_dim,
                                   n_layers=self.rnn_layers, dropout=self.rnn_dropout).to(self.dev)

        self.comb = Combiner(z_dim=self.z_dim, rnn_dim=self.rnn_dim, dynamics_dim=self.dynamics_dim).to(self.dev)

        self.trans = GatedTransition(z_dim=self.z_dim, dynamics_dim=self.dynamics_dim, transition_dim=self.rnn_dim).to(self.dev)

        self.VDSMSeq = VDSM_Seq(rnn_enc=self.rnn_enc,  nc=self.nc, n_e_w=self.n_e_w, id_layers=self.id_layers,
                       dynamics_dim=self.dynamics_dim, input_dim=self.z_dim, comb=self.comb, trans=self.trans,
                       hid_dim=self.rnn_dim, num_layers_rnn=self.rnn_layers, image_dec=self.dec, image_enc=self.enc,
                       imsize=self.imsize, temp_min=self.temp_id_end).to(self.dev)

        self.optim_VDSM_EncDec = torch.optim.Adam([{"params": self.VDSM_EncDec.parameters(), "lr": self.lr_VDSMEncDec}])

        seq_params = [param for name, param in self.VDSMSeq.named_parameters() if 'image' not in name]

        self.optim_VDSM_Seq = torch.optim.Adam([{"params": seq_params, "lr": self.lr_VDSMSeq}])

        self.load_model_opt_sched()
        if self.lr_resume is not None:
            if self.train_VDSMEncDec:
                for g in self.optim_VDSM_EncDec.param_groups:
                    g['lr'] = self.lr_resume
                    print(g['lr'])
            elif self.train_VDSMSeq:
                for g in self.optim_VDSM_Seq.param_groups:
                    g['lr'] = self.lr_resume
                    print(g['lr'])

    def train(self):
        print('Training model...')
        self.vdsm_encdec_loss_fn =Trace_ELBO().differentiable_loss
        self.vdsm_seq_loss_fn = TraceEnum_ELBO(max_plate_nesting=2).differentiable_loss

        for self.current_epoch in range(self.starting_epoch, self.epochs):
            anneal_t = self.anneals_t[self.current_epoch]  # anneals the i.i.d. pose latent (outer))
            anneal_dynamics = self.anneals_dynamics[self.current_epoch]  # anneals the dynamics latent (inner)
            anneal_id = self.anneals_id[self.current_epoch]  # anneals learning the per-sequence ID KL (outer)
            temp_id = self.temps_id[self.current_epoch]   # anneals the temperature of the per-sequence ID simplex dist (outer)

            print('Ann. z', anneal_t, 'Ann. dyn', anneal_dynamics, 'Ann. id', anneal_id, 'id temp', temp_id)
            epoch_loss = torch.tensor([0.]).to(self.dev)

            if self.train_VDSMSeq:
                self.VDSM_EncDec.train()
            else:
                self.VDSM_EncDec.eval()

            if self.train_VDSMSeq:
                self.VDSMSeq.train()
            else:
                self.VDSMSeq.eval()

            for b in range(self.bs_per_epoch):

                if self.train_VDSMEncDec:
                    if self.dataset_name == 'MUG-FED':
                        x, _ = next(iter(self.dataloader_test))
                    elif self.dataset_name == 'sprites':
                        x, _ = next(iter(self.dataloader_train))

                    num_individuals, num_timepoints, pixels = x.view(x.shape[0], x.shape[1],
                                                                     self.imsize ** 2 * self.nc).shape

                    loss = self.vdsm_encdec_loss_fn(model=self.VDSM_EncDec.model, guide=self.VDSM_EncDec.guide,
                                                    x=x.to(self.dev), temp=torch.tensor(temp_id).cuda(),
                                                    anneal_id=anneal_id, anneal_t=anneal_t)

                    assert not torch_isnan(loss)
                    loss.backward()
                    self.optim_VDSM_EncDec.step()

                    self.optim_VDSM_EncDec.zero_grad()
                    epoch_loss += loss

                elif self.train_VDSMSeq:
                    if self.dataset_name == 'MUG-FED':
                        x, _ = next(iter(self.dataloader_test))
                    elif self.dataset_name == 'sprites':
                        _, x = next(iter(self.dataloader_train))
                        x = (x['sprite'] + 1) / 2

                    num_timepoints = x.shape[1]

                    loss = self.vdsm_seq_loss_fn(model=self.VDSMSeq.model, guide=self.VDSMSeq.guide,
                                            anneal_t=torch.tensor(anneal_t), temp_id=torch.tensor(temp_id),
                                            x=x.to(self.dev), anneal_dynamics=anneal_dynamics, anneal_id=anneal_id)

                    assert not torch_isnan(loss)

                    loss.backward(retain_graph=False)
                    print(self.current_epoch, loss)
                    self.optim_VDSM_Seq.step()
                    self.optim_VDSM_Seq.zero_grad()

                    epoch_loss += loss
            epoch_loss = epoch_loss / self.bs_per_epoch / (self.imsize ** 2) / self.nc / \
                                         x.shape[0]
            # epoch_loss = epoch_loss / self.bs_per_epoch / (num_timepoints - 1) / self.bs / (self.imsize**2*self.nc)

            if self.tboard_log:
                self.tboard.add_scalar("total loss", epoch_loss, self.current_epoch)
                self.tboard.add_scalar("id anneal", anneal_id, self.current_epoch)
                self.tboard.add_scalar("zt anneal", anneal_t, self.current_epoch)
                self.tboard.add_scalar("dyanmics anneal", anneal_dynamics, self.current_epoch)
                self.tboard.add_scalar("id temp", temp_id, self.current_epoch)


            if ((self.current_epoch > 0) and (self.current_epoch % self.model_save_interval == 0 )) or ((self.current_epoch+1) == self.epochs):
                self.save_model_opt_sched()

            if (self.current_epoch % self.model_test_interval == 0) or ((self.current_epoch + 1) == self.epochs):
                self.test(self.current_epoch)
            print('epoch', self.current_epoch, 'loss', epoch_loss.item())


    def test(self, epoch=None):
        torch.cuda.empty_cache()
        print('Testing model...')
        self.VDSM_EncDec.eval()
        self.VDSMSeq.eval()

        #### First the blended testing:
        if self.dataset_name == 'MUG-FED':
            test_images, test_y = next(iter(self.dataloader_test))
        elif self.dataset_name == 'sprites':
            _, test_images= next(iter(self.dataloader_test))
            test_images = (test_images['sprite'] + 1) / 2


        rs = []
        ids = []
        targs = []
        blank_image = torch.zeros(self.nc, self.imsize, self.imsize).to(test_images.device)

        n_test_ids = test_images.shape[0]

        for i in range(n_test_ids):
            targ = test_images[i, 0:1]
            targs.append(targ[0])
            r, bc = self.VDSM_EncDec.reconstruct_img(targ.cuda(), temp=self.test_temp_id, ID_spec=None)
            rs.append(r)
            ids.append(bc)

        ids.insert(0, ids[0])
        targs.insert(0, blank_image)
        rs.insert(0, blank_image)

        new_rs = []
        for i in range(n_test_ids + 1):
            for j in range(n_test_ids + 1):
                r, _ = self.VDSM_EncDec.reconstruct_img(targs[i][None].cuda(), temp=self.test_temp_id, ID_spec=ids[j].T)
                new_rs.append(r[0].view(self.nc, self.imsize, self.imsize))

        grid_recon = make_grid(new_rs, nrow=n_test_ids + 1)
        grid_targs_vert = make_grid(targs, nrow=1)
        grid_targs_horiz = make_grid(targs, nrow=self.num_test_ids + 1)
        grid_recon[:, :, :grid_targs_vert.shape[2]] = grid_targs_vert
        grid_recon[:, :grid_targs_vert.shape[2], :] = grid_targs_horiz
        save_image(grid_recon, os.path.join(self.image_path, 'swap_ep_{}_rec.png'.format(epoch)))

        rs = []
        targs = []
        for i in range(n_test_ids):
            targ = test_images[i, 0:1]
            targs.append(targ[0])
            r, bc = self.VDSM_EncDec.reconstruct_img(targ.cuda(), temp=self.test_temp_id, ID_spec=None)
            rs.append(r[0].view(self.nc, self.imsize, self.imsize))
        grid_recon = make_grid(rs, nrow=self.bs)
        grid_targs = make_grid(targs, nrow=self.bs)
        save_image(grid_recon, os.path.join(self.image_path, 'train_{}_rec.png'.format(epoch)))
        save_image(grid_targs, os.path.join(self.image_path, 'train_{}_targ.png'.format(epoch)))

        rs = []
        ids = []
        inputs = []
        targs = []
        blank_image = torch.zeros(self.nc, self.imsize, self.imsize).to(test_images.device)

        for i in range(self.num_test_ids):
            targ = test_images[i, 1:2]
            targs.append(targ[0])
            input = test_images[i, 0:1]
            inputs.append(input[0])
            r, ID = self.VDSMSeq.test_swap(input.cuda(), temp_id=self.temp_id_end, ID_spec=None)
            rs.append(r[0].view(self.nc, self.imsize, self.imsize))
            ids.append(ID)

        ids.insert(0, ids[0])
        targs.insert(0, blank_image)
        inputs.insert(0, blank_image)
        rs.insert(0, blank_image)

        new_rs = []
        for i in range(self.num_test_ids + 1):
            for j in range(self.num_test_ids + 1):
                r, _ = self.VDSMSeq.test_swap(inputs[i][None].cuda(), temp_id=self.temp_id_end, ID_spec=ids[j].T)
                new_rs.append(r[0].view(self.nc, self.imsize, self.imsize))

        grid_recon = make_grid(new_rs, nrow=self.num_test_ids + 1, padding=0)
        grid_targs_vert = make_grid(targs, nrow=1, padding=0)
        grid_targs_horiz = make_grid(targs, nrow=self.num_test_ids + 1, padding=0)
        grid_recon[:, :, :grid_targs_vert.shape[2]] = grid_targs_vert
        grid_recon[:, :grid_targs_vert.shape[2], :] = grid_targs_horiz
        save_image(grid_recon, os.path.join(self.image_path, 'swap_ep_{}_rec.png'.format(epoch)))


        test_images_predict = test_images[:self.num_test_ids, :, :, :]
        self.swap_id_generate_sequence(test_images_predict, epoch)
        num_individuals, num_timepoints, pixels = (test_images_predict.shape[0],
                                                   test_images_predict.shape[1],
                                                   self.imsize ** 2 * self.nc)

        unrav = test_images_predict.view(num_individuals * num_timepoints, self.nc, self.imsize, self.imsize)
        loc, _, ID, ID_scale = self.VDSMSeq.image_enc.forward(unrav.to(self.dev))
        loc = loc.view(num_individuals, num_timepoints, -1)
        ID, _ = self.id_layers(ID, ID_scale)
        ID = torch.mean(ID.view(num_individuals, num_timepoints, -1), 1).unsqueeze(1)
        ID = ID * self.temp_id_end
        ID_exp = torch.exp(ID)
        ID = ID_exp / ID_exp.sum(-1).unsqueeze(-1)
        seq = loc.permute(1, 0, 2)

        futures = self.VDSMSeq.test_sequence(seq, num_timepoints).permute(1, 0, 2)

        # loc = loc[:, 1:]
        recon_gen = torch.zeros(num_individuals, num_timepoints, pixels, device=loc.device)

        for ind in range(num_individuals):
            recon_gen[ind] = self.VDSMSeq.image_dec.forward(futures[ind], ID[ind, 0].unsqueeze(1))

        grid_all = recon_gen.view(num_individuals, num_timepoints, pixels)  # interleave
        grid_seq = make_grid(grid_all.view(-1, self.nc, self.imsize, self.imsize), nrow=num_timepoints)
        save_image(grid_seq, os.path.join(self.image_path, 'seq_{}_rec.png'.format(epoch)))

        grid_targ = make_grid(test_images_predict.view(-1, self.nc, self.imsize, self.imsize), nrow=num_timepoints)
        save_image(grid_targ, os.path.join(self.image_path, 'seq_{}_gt.png'.format(epoch)))

        print('saving GIFs')
        grids = []
        for s in range(num_timepoints):
            grid = make_grid(recon_gen[:, s].view(-1, self.nc, self.imsize, self.imsize),
                             nrow=int(self.num_test_ids//2))
            grids.append(grid)

        grids = torch.stack(grids)

        gif_images = (255 * grids.permute(0, 2, 3, 1).detach().cpu().numpy()).astype('uint8')
        filename = 'vid_{}_gen.gif'.format(epoch)
        if self.dataset_name == 'MUG-FED':
            fps = 16
        else:
            fps = 10
        imageio.mimsave(os.path.join(self.image_path, filename), gif_images, fps=fps)

    def swap_id_generate_sequence(self, test_images, p):
        ID, seq, num_individuals, num_timepoints, pixels = self.extract_id_etc(test_images)

        futures = self.VDSMSeq.test_sequence(seq, num_timepoints).permute(1, 0, 2)

        recon_gen = []

        for ind_1 in range(num_individuals):
            for ind_2 in range(num_individuals):
                recon_gen.append(self.VDSMSeq.image_dec.forward(futures[ind_1], ID[ind_2, 0].unsqueeze(1)))

        recon_gen = torch.stack(recon_gen)
        recon_gen = recon_gen.view(num_individuals ** 2, num_timepoints, self.nc, self.imsize, self.imsize)
        grid_seq = make_grid(recon_gen.view(-1, self.nc, self.imsize, self.imsize), nrow=num_timepoints, padding=0)
        save_image(grid_seq, os.path.join(self.image_path, 'seq_swap_id_{}_gen.png'.format(p)))
        gif_array = recon_gen.permute(1, 0, 2, 3, 4)
        grids = []
        for s in range(num_timepoints):
            grid = make_grid(gif_array[s].view(-1, self.nc, self.imsize, self.imsize),
                             nrow=self.num_test_ids, padding=0)
            grids.append(grid)

        grids = torch.stack(grids)
        print('check', grids.shape, self.num_test_ids)
        gif_images = (255 * grids.permute(0, 2, 3, 1).detach().cpu().numpy()).astype('uint8')
        filename = 'vid_swap_{}_gen.gif'.format(p)
        if self.dataset_name == 'MUG-FED':
            fps = 16
        else:
            fps = 10
        imageio.mimsave(os.path.join(self.image_path, filename), gif_images, fps=fps)
        return

    def extract_id_etc(self, test_images):
        num_individuals, num_timepoints, pixels = (test_images.shape[0],
                                                   test_images.shape[1],
                                                   self.imsize ** 2 * self.nc)

        unrav = test_images.view(num_individuals * num_timepoints, self.nc, self.imsize, self.imsize)
        loc, _, ID, ID_scale = self.VDSMSeq.image_enc.forward(unrav.to(self.dev))
        loc = loc.view(num_individuals, num_timepoints, -1)
        ID, _ = self.id_layers(ID, ID_scale)
        ID = torch.mean(ID.view(num_individuals, num_timepoints, -1), 1).unsqueeze(1)
        ID = ID * self.temp_id_end
        ID_exp = torch.exp(ID)
        ID = ID_exp / ID_exp.sum(-1).unsqueeze(-1)
        seq = loc.permute(1, 0, 2)

        return ID, seq, num_individuals, num_timepoints, pixels

    def annealing(self):
        # import matplotlib.pyplot as plt
        temp = np.arange(0, math.ceil(self.epochs / self.temp_id_frac))
        temps_id = self.temp_id_start + (self.temp_id_end - self.temp_id_start) * \
                     (np.sin(np.pi * temp / (self.epochs / self.temp_id_frac) + np.pi / 2) + 1) / 2
        temps_id = np.flip(np.concatenate((
            np.ones((self.temp_id_frac - 1) * self.epochs // self.temp_id_frac) * self.temp_id_end,
            temps_id)))


        temp = np.arange(0, math.ceil(self.epochs / self.anneal_frac_id))
        anneals_id = self.anneal_start_id + (self.anneal_end_id - self.anneal_start_id) * \
                  (np.sin(np.pi * temp / (self.epochs / self.anneal_frac_id) + np.pi / 2) + 1) / 2
        anneals_id = np.flip(np.concatenate((
                        np.ones((self.anneal_frac_id - 1) * self.epochs // self.anneal_frac_id) * self.anneal_end_id,
                        anneals_id)))

        assert self.anneal_t_midfrac2 > self.anneal_t_midfrac1, '2nd anneal_t frac must be larger than 1st'

        tempx = np.array([0, self.epochs*self.anneal_t_midfrac1, self.epochs*self.anneal_t_midfrac2, self.epochs-1])
        tempy = np.array([self.anneal_start_t, self.anneal_mid_t1, self.anneal_mid_t2, self.anneal_end_t])
        f2 = interp1d(tempx, tempy, kind='quadratic')
        anneals_t = f2(np.linspace(0, self.epochs-1, num=self.epochs, endpoint=True))
        anneals_t = np.clip(anneals_t, a_min=0.005, a_max=80)

        temp = np.arange(0, math.ceil(self.epochs // self.anneal_frac_dynamics))
        anneals_dynamics = self.anneal_start_dynamics + (self.anneal_end_dynamics - self.anneal_start_dynamics) * \
                            (np.sin(np.pi * temp / (self.epochs / self.anneal_frac_dynamics) + np.pi / 2) + 1) / 2
        anneals_dynamics = np.flip(np.concatenate((
            np.ones((
                                self.anneal_frac_dynamics - 1) * self.epochs // self.anneal_frac_dynamics) * self.anneal_end_dynamics,
            anneals_dynamics)))

        return temps_id, anneals_id, anneals_t, anneals_dynamics

    def load_model_opt_sched(self):
        # Load pretrained models (if applicable)
        if self.pretrained_model_VDSMEncDec is not None:
            print('Loading pretrained blended model from epoch: {}'.format(self.pretrained_model_VDSMEncDec))
            checkpoint = torch.load(os.path.join(self.model_save_path,
                                                 '{}_VDSM_EncDec.pth'.format(self.pretrained_model_VDSMEncDec)))
            self.VDSM_EncDec.load_state_dict(checkpoint['model'], strict=False)

            if self.train_VDSMSeq or not self.train_VDSMEncDec:
                self.starting_epoch = checkpoint['epoch'] = 0
            else:
                self.starting_epoch = checkpoint['epoch']
            print('Model loaded.')
        elif self.pretrained_model_VDSMEncDec is None:

            load_file, epoch = check_for_checkpt(self.model_save_path, 'VDSM_EncDec.pth')
            if load_file is None:
                pass
            else:
                checkpoint = torch.load(load_file)
                self.VDSM_EncDec.load_state_dict(checkpoint['model'], strict=False)
                self.optim_VDSM_EncDec.load_state_dict(checkpoint['optimizer'])
                self.starting_epoch = checkpoint['epoch']
                if self.train_VDSMEncDec:
                    self.starting_epoch = epoch

        if self.pretrained_model_VDSMSeq is not None:
            load_file = check_for_checkpt(self.model_save_path, 'VDSM_seq.pth')
            print('Loading pretrained sequence model from epoch: {}'.format(self.pretrained_model_VDSMSeq))
            checkpoint = torch.load(os.path.join(self.model_save_path,
                                                 '{}_VDSM_seq.pth'.format(self.pretrained_model_VDSMSeq)))
            self.VDSMSeq.load_state_dict(checkpoint['model'])
            self.optim_VDSM_Seq.load_state_dict(checkpoint['optimizer'])
            self.starting_epoch = checkpoint['epoch']
            print('Model loaded.')
        elif self.pretrained_model_VDSMSeq is None:
            load_file, epoch = check_for_checkpt(self.model_save_path, 'VDSM_seq.pth')
            if load_file is None:
                pass
            else:
                checkpoint = torch.load(load_file)
                self.VDSMSeq.load_state_dict(checkpoint['model'])
                self.optim_VDSM_Seq.load_state_dict(checkpoint['optimizer'])
                self.starting_epoch = checkpoint['epoch']
                if self.train_VDSMSeq:
                    self.starting_epoch = epoch

        print('Model, opt and sched loaded. Starting from epoch:', self.starting_epoch)

    def save_model_opt_sched(self):
        if self.train_VDSMEncDec:
            blended_checkpoint = {
                'epoch': self.current_epoch + 1,
                'model': self.VDSM_EncDec.state_dict(),
                'optimizer': self.optim_VDSM_EncDec.state_dict()}
            torch.save(blended_checkpoint, os.path.join(self.model_save_path, '{}_VDSM_EncDec.pth'.format(self.current_epoch)))

        elif self.train_VDSMSeq:
            seq_checkpoint = {
                'epoch': self.current_epoch + 1,
                'model': self.VDSMSeq.state_dict(),
                'optimizer': self.optim_VDSM_Seq.state_dict()}
            torch.save(seq_checkpoint, os.path.join(self.model_save_path, '{}_VDSM_seq.pth'.format(self.current_epoch)))

    def load_data(self):
        if self.train_VDSMEncDec:
            pretrain = True
        else:
            pretrain = False

        if self.dataset_name == 'MUG-FED':
            self.dataset_train, self.dataset_test = create_MUG_dataset(self.dataset_dir,
                                                                       self.imsize, self.seq_len, self.seed, pretrain)
            self.dataloader_train = DataLoader(self.dataset_train,
                                               batch_size=self.bs,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=False)

            self.dataloader_test = DataLoader(self.dataset_test,
                                              batch_size=self.num_test_ids,
                                              shuffle=True,
                                              num_workers=0,
                                              pin_memory=False)
        elif self.dataset_name == 'sprites':

            num_train = len(os.listdir(os.path.join(self.dataset_dir, 'train')))
            num_test = len(os.listdir(os.path.join(self.dataset_dir, 'test')))

            self.dataset_train, self.dataset_test = create_sprites_dataset(path=self.dataset_dir,
                                                                           num_train=num_train,
                                                                           num_test=num_test)

            self.dataloader_train = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.bs,
                                                                shuffle=True, num_workers=0,
                                                                pin_memory=False
                                                                )

            self.dataloader_test = torch.utils.data.DataLoader(self.dataset_test, batch_size=self.num_test_ids,
                                                               shuffle=True, num_workers=0,
                                                               pin_memory=False
                                                               )


    def do_seq_evaluation(self, x):
        print('Getting test loss')
        self.VDSMSeq.eval()
        anneal_specifics = self.anneals_specifics[self.current_epoch]
        anneal_t = self.anneals_t[self.current_epoch]
        anneal_dynamics = self.anneals_dynamics[self.current_epoch]
        anneal_id = self.anneals_id[self.current_epoch]
        dynamics_temp = self.dynamics_temps[
            self.current_epoch]
        temp_id = self.temps_id[self.current_epoch]

        test_nll = self.seq_loss_fn(model=self.VDSMSeq.model, guide=self.VDSMSeq.guide, anneal_t=torch.tensor(anneal_t),
                           temp_dynamics=torch.tensor(dynamics_temp),
                           anneal_specifics=anneal_specifics, temp_id=torch.tensor(temp_id),
                           x=x.to(self.dev), anneal_dynamics=anneal_dynamics, anneal_id=anneal_id) / (self.seq_len) / (self.test_repeats)
        self.tboard.add_scalar("test loss", test_nll, self.current_epoch)

        return test_nll