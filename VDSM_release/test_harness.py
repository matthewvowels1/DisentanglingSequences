import imageio
from pyro.util import torch_isnan
from pyro.infer import Trace_ELBO
from dataset import *
from torch.utils.data import DataLoader
import torch
from utilities import check_for_checkpt
import numpy as np
from modules import Enc, Dec, RNN_encoder, Combiner, GatedTransition, ID_Layers
from VDSM import VDSM_EncDec, VDSM_Seq
import os
from torchvision.utils import save_image, make_grid


#  MAKE SURE THE VIDEO FILES ARE EMPTY BEFORE RUNNING THE FID GENERATOR SCRIPT

class Test_Harness(object):
    def __init__(self, config):

        # training settings
        self.RUN = config.RUN
        self.bs = config.bs
        self.test_repeats = 1
        self.bs_per_epoch = config.bs_per_epoch
        self.epochs = config.epochs
        self.seed = config.seed
        self.seq_len = config.seq_len
        self.train_blended = config.train_blended
        self.train_seq = config.train_seq
        self.imsize = config.imsize
        self.model_save_interval = config.model_save_interval
        self.model_test_interval = config.model_test_interval
        self.numsteps = self.bs_per_epoch * self.epochs
        self.dataset_name = config.dataset_name
        self.pretrained_model_blended = config.pretrained_model_blended
        self.pretrained_model_seq= config.pretrained_model_seq
        self.model_save_path = os.path.join(config.model_save_path, config.model, config.RUN)
        self.data_dir = config.data_dir
        self.dataset_dir = os.path.join(self.data_dir, self.dataset_name)
        self.image_path = os.path.join(config.image_path, config.model, config.RUN)
        self.code_path = os.path.join(config.code_path, config.model, config.RUN)
        self.num_test_ids = config.num_test_ids
        self.tboard_log = config.tboard_log
        self.log_dir = os.path.join(config.log_dir, config.model)
        self.nc = config.nc
        self.lr_resume = config.lr_resume

        # blended model settings
        self.likelihood = config.likelihood
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
        self.lr_blended = config.lr_blended

        # seq model params
        self.num_iafs = config.num_iafs
        self.specifics_dim = config.specifics_dim
        self.dynamics_dim = config.dynamics_dim
        self.directions = config.directions
        self.dynamics_temp_end = config.dynamics_temp_end
        self.dynamics_temp_start = config.dynamics_temp_start
        self.dynamics_temp_frac = config.dynamics_temp_frac
        self.rnn_dropout = config.rnn_dropout
        self.anneal_frac_specifics = config.anneal_frac_specifics
        self.anneal_end_specifics = config.anneal_end_specifics
        self.anneal_start_specifics = config.anneal_start_specifics
        self.anneal_frac_dynamics = config.anneal_frac_dynamics
        self.anneal_end_dynamics = config.anneal_end_dynamics
        self.anneal_start_dynamics = config.anneal_start_dynamics
        self.trans_dim = config.trans_dim
        self.iaf_dim = config.iaf_dim
        self.em_dim = config.em_dim
        self.rnn_dim = config.rnn_dim
        self.rnn_z_dim = config.rnn_z_dim
        self.num_iafs = config.num_iafs
        self.rnn_enc_layers = config.rnn_enc_layers
        self.rnn_dec_layers = config.rnn_dec_layers
        self.rnn_dropout = config.rnn_dropout
        self.lr_seq = config.lr_seq
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.clip_norm = config.clip_norm
        self.seq_lrdecay = config.seq_lrdecay
        self.seq_wdecay = config.seq_wdecay
        self.starting_epoch = 0   # this may get overwritten if a checkpoint exists
        self.current_epoch = 0

        self.load_data()

        # Initialize models
        self.dev = torch.device('cuda')
        torch.cuda.empty_cache()
        self.enc = Enc(imsize=self.imsize, z_dim=self.z_dim, nc=self.nc, n_expert_components=self.n_e_w).to(self.dev)
        self.dec = Dec(imsize=self.imsize, z_dim=self.z_dim, nc=self.nc, n_expert_components=self.n_e_w).to(self.dev)
        self.id_layers = ID_Layers(n_e_w=self.n_e_w)
        self.vae = BlendedVAE(enc=self.enc, dec=self.dec, id_layers=self.id_layers, seq_len=self.seq_len,
                              z_dim=self.z_dim, imsize=self.imsize,
                              n_e_w=self.n_e_w, likelihood=self.likelihood, nc=self.nc).to(self.dev)

        self.rnn_enc = RNN_encoder(input_dim=self.z_dim, rnn_dim=self.rnn_dim, output_dim=self.z_dim,
                                   n_layers=self.rnn_enc_layers, directions=self.directions,
                                   dropout=self.rnn_dropout).to(self.dev)
        self.rnn_dec = RNN_decoder(input_dim=self.z_dim, rnn_dim=self.rnn_dim, output_dim=2*self.z_dim,
                                   n_layers=self.rnn_dec_layers, directions=self.directions,
                                   dropout=self.rnn_dropout).to(self.dev)

        self.comb = Combiner(z_dim=self.z_dim, rnn_dim=self.rnn_dim, dynamics_dim=self.dynamics_dim, num_layers_dec=self.rnn_dec_layers,
                             directions=self.directions).to(self.dev)
        self.trans = GatedTransition(z_dim=self.z_dim, dynamics_dim=self.dynamics_dim, transition_dim=self.rnn_dim).to(self.dev)
        self.emit = Emitter(input_dim=self.z_dim, z_dim=self.z_dim, emission_dim=self.rnn_dim).to(self.dev)

        self.dmm = DMM(enc=self.rnn_enc, dec=self.rnn_dec, nc=self.nc, n_e_w=self.n_e_w, id_layers=self.id_layers, comb=self.comb, emit=self.emit, trans=self.trans,
                       directions=self.directions, num_iafs=self.num_iafs, dynamics_dim=self.dynamics_dim, input_dim=self.z_dim,
                       hid_dim=self.rnn_dim, num_layers_enc=self.rnn_enc_layers, image_dec=self.dec, image_enc=self.enc,
                       num_layers_dec=self.rnn_dec_layers, imsize=self.imsize, temp_min=self.dynamics_temp_end,
                       specifics_dim=self.specifics_dim).to(self.dev)

        self.load_model_opt_sched()

    def test(self, epoch=None):
        torch.cuda.empty_cache()
        print('Testing model...')
        self.dmm.eval()

        test_images = None
        labels = None

        for i in range(200):
            #### First the blended testing:
            if self.dataset_name == 'pendulum':
                test_images = next(iter(self.dataloader_train))  # (bs, seq_len, 3, 64, 64)
            elif self.dataset_name == 'MUG-FED':
                test_images, labels = next(iter(self.dataloader_test))
            elif self.dataset_name == 'weizmann' or 'moving_MNIST' in self.dataset_name:
                test_images, labels = next(iter(self.dataloader_test))
                test_images = test_images
            elif self.dataset_name == 'sprites' or self.dataset_name == 'sprites_2':
                test_images, labels = next(iter(self.dataloader_test))
                # item['body'], item['shirt'], item['pant'], item['hair'], item['action'], (item['sprite'] + 1) / 2

            fid_script_root = '/home/matthewvowels/GitHub/video-classification-3d-cnn-pytorch/'
            output_root = '/media/matthewvowels/Storage/video-classification-3d-cnn-pytorch/'

            # self.swap_ids(test_images, i)
            self.generate_GIFs(test_images, i)
            self.generate_codes(test_images, labels, i)
            self.generate_videos(fid_script_root, output_root, test_images, i)
            self.swap_id_generate_sequence(test_images, i)


    def swap_ids(self, test_images, p=0):
        ids = []
        inputs = []
        targs = []
        blank_image = torch.zeros(self.nc, self.imsize, self.imsize).to(test_images.device)
        rand_t = np.random.randint(test_images.shape[1] - 2)
        for i in range(self.num_test_ids):
            targ = test_images[i, rand_t + 1:rand_t + 2]
            targs.append(targ[0])
            input = test_images[i, rand_t:rand_t + 1]
            inputs.append(input[0])
            r, ID = self.dmm.test_swap(input.cuda(), temp_id=self.temp_id_end, ID_spec=None)
            ids.append(ID)

        ids.insert(0, ids[0])
        targs.insert(0, blank_image)
        inputs.insert(0, blank_image)

        new_rs = []
        for i in range(self.num_test_ids + 1):
            for j in range(self.num_test_ids + 1):
                r, _ = self.dmm.test_swap(inputs[i][None].cuda(), temp_id=self.temp_id_end, ID_spec=ids[j].T)
                new_rs.append(r[0].view(self.nc, self.imsize, self.imsize))

        grid_recon = make_grid(new_rs, nrow=self.num_test_ids + 1)
        grid_targs_vert = make_grid(targs, nrow=1)
        grid_targs_horiz = make_grid(targs, nrow=self.num_test_ids + 1)
        grid_recon[:, :, :grid_targs_vert.shape[2]] = grid_targs_vert
        grid_recon[:, :grid_targs_vert.shape[2], :] = grid_targs_horiz
        save_image(grid_recon, os.path.join(self.image_path, 'swap_ep_{}_rec.png'.format(p)))

    def generate_GIFs(self, test_images, p=0):
        num_individuals, num_timepoints, pixels = (test_images.shape[0],
                                                   test_images.shape[1],
                                                   self.imsize ** 2 * self.nc)
        recon_gen = self.generate_sequences(test_images)

        grid_seq = make_grid(recon_gen.view(-1, self.nc, self.imsize, self.imsize), nrow=num_timepoints)
        grid_seq_orig = make_grid(test_images.view(-1, self.nc, self.imsize, self.imsize), nrow=num_timepoints)
        save_image(grid_seq, os.path.join(self.image_path, 'seq_{}_gen.png'.format(p)))
        save_image(grid_seq_orig, os.path.join(self.image_path, 'seq_{}_orig.png'.format(p)))

        print('saving .GIFs')

        grids = []
        for s in range(num_timepoints-1):
            grid = make_grid(recon_gen[:, s].view(-1, self.nc, self.imsize, self.imsize), nrow=int(self.num_test_ids/2))
            grids.append(grid)

        grids = torch.stack(grids)
        gif_images = (255*grids.permute(0, 2, 3, 1).detach().cpu().numpy()).astype('uint8')
        filename = 'vid_{}_gen.gif'.format(p)
        imageio.mimsave(os.path.join(self.image_path, filename), gif_images, fps=10)

    def generate_videos(self, root_dir, output_dir, test_images, p):
        recon_gen = self.generate_sequences(test_images).cpu()
        recon_sampled = self.sample_sequences(test_images).cpu()

        fid_real_dir = os.path.join(output_dir, 'videos/real/')
        fid_fake_dir = os.path.join(output_dir, 'videos/fake/')
        fid_sampled_fake_dir = os.path.join(output_dir, 'videos/sampled/')

        if p == 0:
            for file in os.listdir(os.path.join(fid_real_dir)):
                os.remove(os.path.join(fid_real_dir, file))
            for file in os.listdir(os.path.join(fid_fake_dir)):
                os.remove(os.path.join(fid_fake_dir, file))
            for file in os.listdir(os.path.join(fid_sampled_fake_dir)):
                os.remove(os.path.join(fid_sampled_fake_dir, file))
        if p == 0:
            files = os.listdir(os.path.join(root_dir))
            for file in files:
                if 'input_' in file:
                    print('removing', file)
                    os.remove(os.path.join(root_dir, file))

        self.prepare_videos(batch=test_images, output_root=root_dir, root=root_dir, dir=fid_real_dir, p=p)
        self.prepare_videos(batch=recon_gen, output_root=root_dir, root=root_dir, dir=fid_fake_dir, p=p)
        self.prepare_videos(batch=recon_sampled, output_root=root_dir, root=root_dir, dir=fid_sampled_fake_dir, p=p)

    def generate_codes(self, test_images, labels, p):
        ID, seq, num_individuals, num_timepoints, pixels = self.extract_id_etc(test_images)
        dynamics = self.dmm.return_dynamics(seq).detach().cpu().numpy()
        recon_gen = self.generate_sequences(test_images).detach().cpu().numpy()
        recon_sampled = self.sample_sequences(test_images).cpu().detach().numpy()

        np.savez(os.path.join(self.code_path, 'id_{}.npz'.format(p)), ID.detach().cpu().numpy())
        if labels is not None:
            np.savez(os.path.join(self.code_path, 'labels_{}.npz'.format(p)), labels)
        np.savez(os.path.join(self.code_path, 'dynamics_{}.npz'.format(p)), dynamics)
        np.savez(os.path.join(self.code_path, 'test_images_{}.npz'.format(p)), test_images.detach().cpu().numpy())
        np.savez(os.path.join(self.code_path, 'recon_images_{}.npz'.format(p)), recon_gen)
        np.savez(os.path.join(self.code_path, 'sampled_images_{}.npz'.format(p)), recon_sampled)

    def extract_id_etc(self, test_images):
        num_individuals, num_timepoints, pixels = (test_images.shape[0],
                                                   test_images.shape[1],
                                                   self.imsize ** 2 * self.nc)

        unrav = test_images.view(num_individuals * num_timepoints, self.nc, self.imsize, self.imsize)
        loc, _, ID, ID_scale = self.dmm.image_enc.forward(unrav.to(self.dev))
        loc = loc.view(num_individuals, num_timepoints, -1)
        ID, _ = self.id_layers(ID, ID_scale)
        ID = torch.mean(ID.view(num_individuals, num_timepoints, -1), 1).unsqueeze(1)
        ID = ID * self.temp_id_end
        ID_exp = torch.exp(ID)
        ID = ID_exp / ID_exp.sum(-1).unsqueeze(-1)
        seq = loc.permute(1, 0, 2)

        return ID, seq, num_individuals, num_timepoints, pixels

    def prepare_videos(self, batch, output_root, root, dir, p):

        filenames = []
        for i, seq in enumerate(batch):
            filename = 'vid_{}_{}_gen.mp4'.format(p, i)
            filenames.append(filename)
            if self.nc == 1:
                seq = seq.repeat(1, 3, 1, 1)
            seq = resize_tensor(seq, 112, 112).permute(0, 2, 3, 1).detach().cpu().numpy() * 255
            torchvision.io.write_video(filename=os.path.join(output_root, dir, filename),
                                       video_array=seq,
                                       fps=10)


    def generate_sequences(self, test_images):
        ID, seq, num_individuals, num_timepoints, pixels = self.extract_id_etc(test_images)

        futures = self.dmm.test_sequence(seq, num_timepoints).permute(1, 0, 2)

        recon_gen = torch.zeros(num_individuals, num_timepoints, pixels, device=test_images.device)

        for ind in range(num_individuals):
            recon_gen[ind] = self.dmm.image_dec.forward(futures[ind], ID[ind, 0].unsqueeze(1))
        recon_gen = recon_gen.view(num_individuals, num_timepoints, self.nc, self.imsize, self.imsize)
        return recon_gen

    def extract_id_etc(self, test_images):
        num_individuals, num_timepoints, pixels = (test_images.shape[0],
                                                   test_images.shape[1],
                                                   self.imsize ** 2 * self.nc)

        unrav = test_images.view(num_individuals * num_timepoints, self.nc, self.imsize, self.imsize)
        loc, _, ID, ID_scale = self.dmm.image_enc.forward(unrav.to(self.dev))
        loc = loc.view(num_individuals, num_timepoints, -1)
        ID, _ = self.id_layers(ID, ID_scale)
        ID = torch.mean(ID.view(num_individuals, num_timepoints, -1), 1).unsqueeze(1)
        ID = ID * self.temp_id_end
        ID_exp = torch.exp(ID)
        ID = ID_exp / ID_exp.sum(-1).unsqueeze(-1)
        seq = loc.permute(1, 0, 2)

        return ID, seq, num_individuals, num_timepoints, pixels

    def sample_sequences(self, test_images):
        recon_gen = self.dmm.sample_sequence(test_images)
        return recon_gen

    def swap_id_generate_sequence(self, test_images, p):
        ID, seq, num_individuals, num_timepoints, pixels = self.extract_id_etc(test_images)

        futures = self.dmm.test_sequence(seq, num_timepoints).permute(1, 0, 2)

        recon_gen = []

        for ind_1 in range(num_individuals):
            for ind_2 in range(num_individuals):
                recon_gen.append(self.dmm.image_dec.forward(futures[ind_1], ID[ind_2, 0].unsqueeze(1)))

        recon_gen = torch.stack(recon_gen)
        recon_gen = recon_gen.view(num_individuals**2, num_timepoints, self.nc, self.imsize, self.imsize)
        grid_seq = make_grid(recon_gen.view(-1, self.nc, self.imsize, self.imsize), nrow=num_timepoints)
        save_image(grid_seq, os.path.join(self.image_path, 'seq_swap_id_{}_gen.png'.format(p)))
        return

    def load_model_opt_sched(self):

        # Load pretrained models (if applicable)
        if self.pretrained_model_blended is not None:
            print('Loading pretrained blended model from epoch: {}'.format(self.pretrained_model_blended))
            checkpoint = torch.load(os.path.join(self.model_save_path,
                                                 '{}_VAE.pth'.format(self.pretrained_model_blended)))
            self.vae.load_state_dict(checkpoint['model'], strict=False)


            print('Model loaded.')
        elif self.pretrained_model_blended is None:
            load_file, epoch = check_for_checkpt(self.model_save_path, 'VAE.pth')
            if load_file is None:
                pass
            else:
                checkpoint = torch.load(load_file)
                self.vae.load_state_dict(checkpoint['model'], strict=False)

        if self.pretrained_model_seq is not None:
            load_file = check_for_checkpt(self.model_save_path, 'DMM.pth')
            print('Loading pretrained sequence model from epoch: {}'.format(self.pretrained_model_seq))
            checkpoint = torch.load(os.path.join(self.model_save_path,
                                                 '{}_DMM.pth'.format(self.pretrained_model_seq)))
            self.dmm.load_state_dict(checkpoint['model'])
            print('Model loaded.')
        elif self.pretrained_model_seq is None:
            load_file, epoch = check_for_checkpt(self.model_save_path, 'DMM.pth')
            if load_file is None:
                pass
            else:
                checkpoint = torch.load(load_file)
                self.dmm.load_state_dict(checkpoint['model'])

        print('Model, opt and sched loaded.')

    def load_data(self):
        if self.dataset_name == 'pendulum':
            self.dataset_train, self.dataset_test = create_pendulum_dataset(self.dataset_dir, self.imsize, self.seq_len)
            self.dataloader_train = DataLoader(self.dataset_train,
                                               batch_size=self.bs,
                                               shuffle=True,
                                               num_workers=1,
                                               pin_memory=torch.cuda.is_available())
            self.dataloader_test= DataLoader(self.dataset_test,
                                               batch_size=self.num_test_ids,
                                               shuffle=True,
                                               num_workers=1,
                                               pin_memory=torch.cuda.is_available())
        elif self.dataset_name == 'MUG-FED':
            self.dataset_train, self.dataset_test = create_MUG_dataset(self.dataset_dir,
                                                                       self.imsize, self.seq_len, self.seed)
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

        elif self.dataset_name == 'weizmann':
            def my_collate(batch):
                remove_actions = [1, 2, 3, 7, 8, 9, 10, 14, 15]

                batch = list(filter(lambda x: (x[-1][1] not in remove_actions), batch))
                print(len(batch))
                return dataloader.default_collate(batch)

            self.dataset_train, self.dataset_test = create_weizmann_dataset(self.dataset_dir,

                                                                            self.imsize, self.seq_len, self.seed)

            self.dataloader_train = DataLoader(self.dataset_train,
                                               batch_size=self.bs,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=False, collate_fn=my_collate)

            self.dataloader_test = DataLoader(self.dataset_test,
                                              batch_size=self.num_test_ids,
                                              shuffle=True,
                                              num_workers=0,
                                              pin_memory=False, collate_fn=my_collate)

        elif self.dataset_name == 'sprites' or self.dataset_name == 'sprites_2':

            # https://github.com/yatindandi/Disentangled-Sequential-Autoencoder/blob/master/trainer.py

            self.dataset_train, self.dataset_test = create_sprites_dataset(path=self.dataset_dir,
                                                                           num_train=6723,
                                                                           num_test=837)

            self.dataloader_train = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.bs,
                                                                shuffle=True, num_workers=0,
                                                                pin_memory=False
                                                                )

            self.dataloader_test = torch.utils.data.DataLoader(self.dataset_test, batch_size=self.num_test_ids,
                                                               shuffle=True, num_workers=0,
                                                               pin_memory=False
                                                               )

        elif 'moving_MNIST' in self.dataset_name:
            type_ = int(self.dataset_name[-1])
            # https://github.com/yatindandi/Disentangled-Sequential-Autoencoder/blob/master/trainer.py
            self.dataset_train, self.dataset_test = create_movingMNIST_dataset(folder=self.dataset_dir, imsize=self.imsize,
                                                                               slice_length=self.seq_len, type_=type_)
            self.dataloader_train = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.bs,
                                                                shuffle=True, num_workers=0,
                                                                pin_memory=False
                                                                )

            self.dataloader_test = torch.utils.data.DataLoader(self.dataset_test, batch_size=self.num_test_ids,
                                                               shuffle=True, num_workers=0,
                                                               pin_memory=False
                                                               )
