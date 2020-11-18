import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro import poutine


class VDSM_EncDec(nn.Module):
    def __init__(self, enc, dec, z_dim, id_layers, imsize, n_e_w, seq_len, likelihood, nc):
        super(VDSM_EncDec, self).__init__()
        self.enc = enc
        self.dec = dec
        self.id_layers = id_layers
        self.seq_len = seq_len
        self.z_dim = z_dim
        self.n_e_w = n_e_w
        self.imsize = imsize
        self.nc = nc
        self.likelihood = likelihood
        self.cuda()
        self.ID_loc_layer = torch.nn.Linear(self.n_e_w, self.n_e_w)
        self.ID_scale_layer = torch.nn.Linear(self.n_e_w, self.n_e_w)

    def model(self, x, temp=1, anneal_id=1.0, anneal_t=1.0):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        pyro.module('BlendedVAE', self)
        num_individuals, num_timepoints, pixels = x.view(x.shape[0], x.shape[1], self.imsize ** 2 * self.nc).shape

        id_plate = pyro.plate("individuals", num_individuals, dim=-2)
        time_plate = pyro.plate("time", num_timepoints, dim=-1)

        out_all = torch.zeros(num_individuals, num_timepoints, pixels, device=x.device)

        with id_plate:
            IDdist = dist.Normal(0, 1 / self.n_e_w).expand([self.n_e_w]).to_event(1)

            with poutine.scale(scale=anneal_id):
                ID = pyro.sample("ID", IDdist).to(x.device) * temp
            ID_exp = torch.exp(ID)
            ID = ID_exp / ID_exp.sum(-1).unsqueeze(-1)

            with time_plate:
                zdist = dist.Normal(0, 1).expand([self.z_dim]).to_event(1)
                with poutine.scale(scale=anneal_t):
                    z = pyro.sample("z", zdist).to(x.device)

                # for each individual and each timepoint, generate reconstructions through NN
                for ind in range(num_individuals):
                    seq = self.dec.forward(z[ind], ID[ind, 0].unsqueeze(1))
                    out_all[ind] = seq.view(-1, pixels)

                if self.likelihood == 'Bernoulli':
                    f = dist.Bernoulli(out_all, validate_args=False).to_event(1)
                elif self.likelihood == 'Normal':
                    f = dist.Bernoulli(out_all, 0.1, validate_args=False).to_event(1)
                elif self.likelihood == 'Laplace':
                    f = dist.Laplace(out_all, 0.1, validate_args=False).to_event(1)
                x = x.view(num_individuals, num_timepoints, pixels)
                pyro.sample('obs', f, obs=x)
        return seq

    def guide(self, x, temp=1, anneal_id=1.0, anneal_t=1.0):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        pyro.module('BlendedVAE', self)
        num_individuals, num_timepoints, pixels = x.view(x.shape[0], x.shape[1], self.imsize ** 2 * self.nc).shape

        id_plate = pyro.plate("individuals", num_individuals, dim=-2)
        time_plate = pyro.plate("time", num_timepoints, dim=-1)

        # pass all sequences in and generate mean, sigma and ID
        x = x.view(num_individuals * num_timepoints, self.nc, self.imsize, self.imsize)
        z_loc, z_scale, ID_loc, ID_scale = self.enc(x)
        ID_loc, ID_scale = self.id_layers(ID_loc, ID_scale)  # extra trainable layer
        ID_loc = torch.mean(ID_loc.view(num_individuals, num_timepoints, -1), 1).unsqueeze(1)
        ID_scale = torch.mean(ID_scale.view(num_individuals, num_timepoints, -1), 1).unsqueeze(1)
        z_loc = z_loc.view(num_individuals, num_timepoints, -1)
        z_scale = z_scale.view(num_individuals, num_timepoints, -1)

        # within the individuals plate:
        with id_plate:
            IDdist = dist.Normal(ID_loc, ID_scale).to_event(1)
            with poutine.scale(scale=anneal_id):
                ID = pyro.sample('ID', IDdist) * temp

            # within the individuals and timepoint plates:
            with time_plate:
                zdist = dist.Normal(z_loc, z_scale).to_event(1)
                with poutine.scale(scale=anneal_t):
                    z = pyro.sample('z', zdist)
        return z_loc, z_scale

    def reconstruct_img(self, x, temp, ID_spec=None):
        z_loc, z_scale, ID, ID_scale = self.enc.forward(x)
        ID, _ = self.id_layers(ID, ID_scale)
        if ID_spec is None:
            ID = ID * temp
            ID_exp = torch.exp(ID)
            ID = ID_exp / ID_exp.sum(-1).unsqueeze(-1)
            loc_img = self.dec.forward(z_loc, ID.T)
            return loc_img, ID
        else:
            loc_img = self.dec.forward(z_loc, ID_spec)
            return loc_img, ID_spec



class VDSM_Seq(nn.Module):

    def __init__(self, rnn_enc, image_enc, image_dec, n_e_w, id_layers, comb, trans, temp_min, imsize=64, dynamics_dim=10,
                 input_dim=10, num_layers_rnn=2, nc=3, hid_dim=512):
        super().__init__()
        # instantiate PyTorch modules used in the model and guide below

        self.imsize = imsize  # size of image (e.g. 64)
        self.input_dim = input_dim  # this is actually zt dim (the input dim of the rnn enc/dec)
        self.hid_dim = hid_dim  # this is the dim of the rnn
        self.nc = nc  # num image channels
        self.id_layers = id_layers   # static factor layers
        self.n_e_w = n_e_w  # num identities/static factor dim
        self.temp_min = temp_min  # temperature on static factors
        self.num_layers_rnn = num_layers_rnn  # temperature on static factors
        self.dynamics_dim = dynamics_dim  # dimensionality of dynamics/action factors
        self.image_dec = image_dec  # pretrained VAE enc
        self.image_enc = image_enc  # pretrained MoE dec
        self.seq2seq_enc = rnn_enc
        self.seq2seq_dec = nn.LSTM(self.hid_dim, self.hid_dim, num_layers=self.num_layers_rnn, bidirectional=False)
        self.comb = comb   # combiner function
        self.transitions = trans  # transition function
        self.h_0_enc = nn.Parameter(torch.zeros(1, 1, self.hid_dim), requires_grad=True)
        self.c_0_enc = nn.Parameter(torch.zeros(1, 1, self.hid_dim), requires_grad=True)
        self.c_0_dec = nn.Parameter(torch.zeros(1, 1, self.hid_dim), requires_grad=True)
        self.z_q_0 = nn.Parameter(torch.zeros(self.input_dim), requires_grad=True)
        self.dec_inp_0 = nn.Parameter(torch.zeros(self.hid_dim), requires_grad=True)
        self.cats = torch.nn.Linear(self.hid_dim * 2 * self.num_layers_rnn,
                                    2 * self.dynamics_dim)  # encodes WHICH action (i.e. which dynamics)
        self.dz_to_dec_h = torch.nn.Linear(self.dynamics_dim, self.hid_dim * self.num_layers_rnn)
        self.dz_to_dec_c = torch.nn.Linear(self.dynamics_dim, self.hid_dim * self.num_layers_rnn)
        self.act = nn.LeakyReLU()
        self.softplus = nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()


    def model(self, x, temp_id=None, anneal_id=None, anneal_t=None, anneal_dynamics=None):
        pyro.module('vdsm_seq', self)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        bs, seq_len, pixels = x.view(x.shape[0], x.shape[1], self.imsize ** 2 * self.nc).shape

        id_prior = x.new_zeros([bs, self.n_e_w])
        dynamics_prior = x.new_zeros([bs, self.dynamics_dim])

        # sample dynamics and identity from prior
        with pyro.plate('ID_plate', bs):
            IDdist = dist.Normal(id_prior, 1 / self.n_e_w).to_event(1)
            dz_dist = dist.Normal(dynamics_prior, 1.0 / self.dynamics_dim).to_event(1)
            with poutine.scale(scale=anneal_id):
                ID = pyro.sample("ID", IDdist).to(x.device) * temp_id  # static factors
            with poutine.scale(scale=anneal_dynamics):
                dz = pyro.sample("dz", dz_dist)  # dynamics factors
            ID_exp = torch.exp(ID)
            ID = ID_exp / ID_exp.sum(-1).unsqueeze(-1)

        zs = torch.zeros(bs, seq_len, self.input_dim)

        for i in pyro.plate('batch_loop', bs):
            z_prev = pyro.sample('z_{}_0'.format(i), dist.Normal(torch.zeros(self.input_dim), 1).to_event(1))
            zs[i, 0] = z_prev
            for t in pyro.markov(range(1, seq_len)):
                z_loc, z_scale = self.transitions(z_prev, dz[None, i].expand(1, self.dynamics_dim))
                z_dist = dist.Normal(z_loc, z_scale).to_event(1)
                with poutine.scale(scale=anneal_t):
                    z = pyro.sample('z_{}_{}'.format(i, t), z_dist)
                zs[i, t] = z
                z_prev = z

        x = torch.flatten(x, 2)
        recon = torch.zeros(bs, seq_len, self.imsize**2 * self.nc, device=x.device)

        for ind in range(bs):
            recon[ind] = self.image_dec(zs[ind], ID[ind].unsqueeze(1))

        with pyro.plate('timepoints_ims', seq_len):
            with pyro.plate('inds_ims', bs):
                image_d = dist.Bernoulli(recon).to_event(1)
                with poutine.scale(scale=1.0):
                    pyro.sample('images', image_d, obs=x)


    def guide(self, x, temp_id=None, anneal_id=None, anneal_t=None, anneal_dynamics=None):
        pyro.module('vdsm_seq', self)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        bs, seq_len, pixels = x.view(x.shape[0], x.shape[1], self.imsize ** 2 * self.nc).shape
        h_0_enc = self.h_0_enc.expand(2 * self.num_layers_rnn, bs, self.hid_dim).contiguous()
        c_0_enc = self.c_0_enc.expand(2 * self.num_layers_rnn, bs, self.hid_dim).contiguous()
        z_prev_ = self.z_q_0.expand(bs, 1, -1)  # z0
        dec_inp_0 = self.dec_inp_0.expand(bs, 1, -1)
        x = x.view(bs * seq_len, self.nc, self.imsize, self.imsize)

        pre_z, _, ID_loc, ID_scale = self.image_enc(x)
        ID_loc, ID_scale = self.id_layers(ID_loc, ID_scale)  # extra trainable layer
        ID_loc = torch.mean(ID_loc.view(bs, seq_len, -1), 1).unsqueeze(1)[:, 0]
        ID_scale = torch.mean(ID_scale.view(bs, seq_len, -1), 1).unsqueeze(1)[:, 0]
        pre_z = pre_z.view(bs, seq_len, -1)

        # from https://github.com/yatindandi/Disentangled-Sequential-Autoencoder/blob/master/model.py self.encode_f
        sequence = pre_z.permute(1, 0, 2)
        _, h, _, rnn_enc_raw, out = self.seq2seq_enc(sequence, h_0_enc, c_0_enc)
        h = h.permute(1, 0, 2).contiguous()
        h = h.view(bs, self.hid_dim * 2 * self.num_layers_rnn)

        d_params = self.cats(h)
        dz_loc = self.act(d_params[:, :self.dynamics_dim])
        dz_scale = self.softplus(d_params[:, self.dynamics_dim:])

        # infer dynamics and identity from data
        with pyro.plate('ID_plate', bs):
            IDdist = dist.Normal(ID_loc, ID_scale).to_event(1)
            dz_dist = dist.Normal(dz_loc, dz_scale).to_event(1)
            with poutine.scale(scale=anneal_id):
                ID = pyro.sample('ID', IDdist) * temp_id  # static factors
            with poutine.scale(scale=anneal_dynamics):
                dz = pyro.sample("dz", dz_dist)  # dynamics z

        h_dec = self.dz_to_dec_h(dz).view(-1, self.num_layers_rnn, self.hid_dim).permute(1, 0, 2)
        c_dec = self.dz_to_dec_c(dz).view(-1, self.num_layers_rnn, self.hid_dim).permute(1, 0, 2)

        for i in pyro.plate('batch_loop', bs):
            dec_inp = dec_inp_0[None, i].contiguous()
            z_prev = z_prev_[None, i].contiguous()
            h = h_dec[:, None, i]
            c = c_dec[:, None, i]
            dz_dec = dz[None, i, None, :]
            for t in pyro.markov(range(seq_len)):
                dec_inp, (h, c) = self.seq2seq_dec(dec_inp, (h, c))
                z_loc, z_scale = self.comb(z_prev, dec_inp, dz_dec)
                z_dist = dist.Normal(z_loc[0], z_scale[0]).to_event(1)
                with poutine.scale(scale=anneal_t):
                    z = pyro.sample('z_{}_{}'.format(i, t), z_dist)
                z_prev = z.view(1, 1, -1)

    def test_sequence(self, sequence, seq_len):
        print('Taking a sequence and reconstructing it ')
        bs = sequence.shape[1]
        h_0_enc = self.h_0_enc.expand(2 * self.num_layers_rnn, bs, self.hid_dim).contiguous()
        c_0_enc = self.c_0_enc.expand(2 * self.num_layers_rnn, bs, self.hid_dim).contiguous()
        z_prev_ = self.z_q_0.expand(bs, 1, -1)  # z0
        dec_inp_0 = self.dec_inp_0.expand(bs, 1, -1)
        _, h, _, rnn_enc_raw, out = self.seq2seq_enc(sequence, h_0_enc, c_0_enc)
        h = h.permute(1, 0, 2).contiguous()
        h = h.view(bs, self.hid_dim * 2 * self.num_layers_rnn)
        d_params = self.cats(h)
        dz_loc = self.act(d_params[:, :self.dynamics_dim])
        h_dec = self.dz_to_dec_h(dz_loc).view(-1, self.num_layers_rnn, self.hid_dim).permute(1, 0, 2)
        c_dec = self.dz_to_dec_c(dz_loc).view(-1, self.num_layers_rnn, self.hid_dim).permute(1, 0, 2)

        futures = torch.zeros(seq_len, bs, self.input_dim, device=sequence.device)

        for i in range(bs):
            dec_inp = dec_inp_0[None, i].contiguous()
            z_prev = z_prev_[None, i].contiguous()
            h = h_dec[:, None, i]
            c = c_dec[:, None, i]
            dz_dec = dz_loc[None, i, None, :]
            for t in range(seq_len):
                dec_inp, (h, c) = self.seq2seq_dec(dec_inp, (h, c))
                z_loc, z_scale = self.comb(z_prev, dec_inp, dz_dec)
                futures[t, i] = z_loc[0]
                z_prev = z_loc[0].view(1, 1, -1)
        return futures

    def test_swap(self, x, temp_id, ID_spec=None):
        # encode image x
        z_loc, _, ID, ID_scale = self.image_enc.forward(x)
        ID, _ = self.id_layers(ID, ID_scale)
        z_prev_ = self.z_q_0.expand(1, 1, -1)
        dec_inp_0 = self.dec_inp_0.expand(1, 1, -1)
        h_0_enc = self.h_0_enc.expand(2 * self.num_layers_rnn, 1, self.hid_dim).contiguous()
        c_0_enc = self.c_0_enc.expand(2 * self.num_layers_rnn, 1, self.hid_dim).contiguous()

        sequence = z_loc.reshape(1, 1, -1)
        _, h, _, rnn_enc_raw, out = self.seq2seq_enc(sequence, h_0_enc, c_0_enc)
        h = h.permute(1, 0, 2).contiguous()
        h = h.view(1, self.hid_dim * 2 * self.num_layers_rnn)

        d_params = self.cats(h)

        dz_loc = self.act(d_params[:, :self.dynamics_dim])  # this is the loc for dz
        h_dec = self.dz_to_dec_h(dz_loc).view(-1, self.num_layers_rnn, self.hid_dim).permute(1, 0, 2)
        c_dec = self.dz_to_dec_c(dz_loc).view(-1, self.num_layers_rnn, self.hid_dim).permute(1, 0, 2)

        dec_inp, (_, _) = self.seq2seq_dec(dec_inp_0, (h_dec, c_dec))
        dz_dec = dz_loc.expand(1, 1, self.dynamics_dim)
        z_loc, z_scale = self.comb(z_prev_, dec_inp, dz_dec)
        z_loc, _ = self.transitions(z_loc, dz_dec)
        z_loc = z_loc[0, :, :]

        if ID_spec is None:
            ID = ID * temp_id
            ID_exp = torch.exp(ID)
            ID = ID_exp / ID_exp.sum(-1).unsqueeze(-1)
            loc_img = self.image_dec.forward(z_loc, ID.T)
            return loc_img, ID
        else:
            loc_img = self.image_dec.forward(z_loc, ID_spec)
            return loc_img, ID_spec


    def return_dynamics(self, sequence):
        print('Taking a sequence and reconstructing it ')
        bs = sequence.shape[1]
        h_0_enc = self.h_0_enc.expand(2 * self.num_layers_rnn, bs, self.hid_dim).contiguous()
        c_0_enc = self.c_0_enc.expand(2 * self.num_layers_rnn, bs, self.hid_dim).contiguous()

        _, h, _, rnn_enc_raw, out = self.seq2seq_enc(sequence, h_0_enc, c_0_enc)
        h = h.permute(1, 0, 2).contiguous()
        h = h.view(bs, self.hid_dim * 2 * self.num_layers_rnn)

        d_params = self.cats(h)
        dz_loc = self.act(d_params[:, :self.dynamics_dim])  # this is the loc for dz
        return dz_loc

    def sample_sequence(self, test_images):
        test_images = test_images.to('cuda')
        bs, seq_len, _, _, _ = test_images.shape
        pixels = self.imsize**2 * self.nc

        id_prior = test_images.new_zeros([bs, self.n_e_w])
        dynamics_prior = test_images.new_zeros([bs, self.dynamics_dim])

        # sample dynamics and identity from prior
        with pyro.plate('ID_plate', bs):
            IDdist = dist.Normal(id_prior, 1 / self.n_e_w).to_event(1)
            dz_dist = dist.Normal(dynamics_prior, 1.0 / self.dynamics_dim).to_event(1)
            ID = pyro.sample("ID", IDdist).to(test_images.device) * self.temp_min  # static factors
            dz = pyro.sample("dz", dz_dist)  # dynamics factors
            ID_exp = torch.exp(ID)
            ID = ID_exp / ID_exp.sum(-1).unsqueeze(-1)

        zs = torch.zeros(bs, seq_len, self.input_dim).to(test_images.device)

        for i in pyro.plate('batch_loop', bs):
            z_prev = pyro.sample('z_{}_0'.format(i), dist.Normal(torch.zeros(self.input_dim), 1).to_event(1)).to(test_images.device)
            zs[i, 0] = z_prev
            for t in pyro.markov(range(1, seq_len)):
                z_loc, z_scale = self.transitions(z_prev.reshape(1, -1), dz[None, i].expand(1, self.dynamics_dim))
                z_dist = dist.Normal(z_loc, z_scale).to_event(1)
                z = pyro.sample('z_{}_{}'.format(i, t), z_dist)
                zs[i, t] = z[0]
                z_prev = z

        recon = torch.zeros(bs, seq_len, self.imsize ** 2 * self.nc, device=test_images.device)

        for ind in range(bs):
            recon[ind] = self.image_dec(zs[ind], ID[ind].unsqueeze(1))

        return recon.view(bs, seq_len, self.nc, self.imsize, self.imsize)