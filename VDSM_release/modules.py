import torch
import torch.nn.functional  as F
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import antialiased_cnns
from utilities import View, kaiming_init, normal_init


class Enc(nn.Module):
    """ Generate latent parameters for CUB image data. """

    def __init__(self, imsize=64, z_dim=100, base_channels=32, nc=3, n_expert_components=5):
        super(Enc, self).__init__()

        self.imsize = imsize
        self.n_expert_components = n_expert_components
        self.z_dim = z_dim
        self.nc = nc

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.nc, out_channels=32, kernel_size=4, stride=1, padding=1),  # B,  32, 32, 32
            nn.LeakyReLU(True),
            antialiased_cnns.BlurPool(32, filt_size=4, stride=2),
            nn.Conv2d(32, 32, 4, 1, 1),  # B,  32, 16, 16
            nn.LeakyReLU(True),
            antialiased_cnns.BlurPool(32, filt_size=4, stride=2),
            nn.Conv2d(32, 64, 4, 1, 1),  # B,  64,  8,  8
            nn.LeakyReLU(True),
            antialiased_cnns.BlurPool(64, filt_size=4, stride=2),
            nn.Conv2d(64, 64, 4, 1, 1),  # B,  64,  4,  4
            nn.LeakyReLU(True),
            antialiased_cnns.BlurPool(64, filt_size=4, stride=2),
            nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
            nn.LeakyReLU(True),
            View((-1, 256 * 1 * 1)))  # B, 256)

        self.muvar = nn.Linear(256, z_dim * 2)
        self.BC_1 = nn.Linear(256, 128)
        self.BC_2 = nn.Linear(128, 2 * self.n_expert_components)
        # self.weight_init()
        self.act = nn.LeakyReLU()
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def weight_init(self):
        for block in self.encoder:
            kaiming_init(block)

    def forward(self, x):
        last_layer_out = self.encoder(x)
        params = self.muvar(last_layer_out)
        loc = params[:, :self.z_dim]
        scale = self.softplus(params[:, self.z_dim:])
        BC = self.BC_2(self.act(self.BC_1(last_layer_out)))
        BC_loc = BC[:, :self.n_expert_components]
        BC_scale = self.softplus(BC[:, self.n_expert_components:])
        return loc, scale, BC_loc , BC_scale


class Dec(nn.Module):
    """ Generate an image given a sample from the latent space. """

    def __init__(self, imsize=5, z_dim=100, nc=3, n_expert_components=5):
        super(Dec, self).__init__()
        self.imsize = imsize
        self.z_dim = z_dim
        self.n_e_w = n_expert_components
        self.conv_dim = 64
        self.nc = nc

        # weights
        self.EW1 = nn.Parameter(torch.Tensor(self.n_e_w, self.z_dim + self.n_e_w, 512, 4, 4), requires_grad=True)
        self.EW2 = nn.Parameter(torch.Tensor(self.n_e_w, 512, 256, 4, 4), requires_grad=True)
        self.EW3 = nn.Parameter(torch.Tensor(self.n_e_w, 256, 128, 4, 4), requires_grad=True)
        self.EW4 = nn.Parameter(torch.Tensor(self.n_e_w, 128, 64, 4, 4), requires_grad=True)
        self.EW5 = nn.Parameter(torch.Tensor(self.n_e_w, 64, self.nc, 4, 4), requires_grad=True)
        # biases
        self.EB1 = nn.Parameter(torch.Tensor(self.n_e_w, 512), requires_grad=True)
        self.EB2 = nn.Parameter(torch.Tensor(self.n_e_w, 256), requires_grad=True)
        self.EB3 = nn.Parameter(torch.Tensor(self.n_e_w, 128), requires_grad=True)
        self.EB4 = nn.Parameter(torch.Tensor(self.n_e_w, 64), requires_grad=True)
        self.EB5 = nn.Parameter(torch.Tensor(self.n_e_w, self.nc), requires_grad=True)

        self.act = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.EW1, a=np.sqrt(5))
        init.kaiming_uniform_(self.EW2, a=np.sqrt(5))
        init.kaiming_uniform_(self.EW3, a=np.sqrt(5))
        init.kaiming_uniform_(self.EW4, a=np.sqrt(5))
        init.kaiming_uniform_(self.EW5, a=np.sqrt(5))
        init.zeros_(self.EB1)
        init.zeros_(self.EB2)
        init.zeros_(self.EB3)
        init.zeros_(self.EB4)
        init.zeros_(self.EB5)

    def forward(self, z, BC):
        BC_concat = BC.permute(1, 0).repeat(z.shape[0], 1)
        BC_w = BC.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        BC_b = BC
        z = torch.cat((z, BC_concat), 1)
        z = z.unsqueeze(-1).unsqueeze(-1)
        z = z.view(-1, *z.size()[-3:])

        out = self.act((torch.nn.functional.conv_transpose2d(z, torch.sum(BC_w * self.EW1, dim=0),
                                                             torch.sum(BC_b * self.EB1, dim=0))))

        out = self.act((torch.nn.functional.conv_transpose2d(out, torch.sum(BC_w * self.EW2, dim=0),
                                                             torch.sum(BC_b * self.EB2, dim=0), 2, 1)))

        out = self.act((torch.nn.functional.conv_transpose2d(out, torch.sum(BC_w * self.EW3, dim=0),
                                                             torch.sum(BC_b * self.EB3, dim=0), 2, 1)))

        out = self.act((torch.nn.functional.conv_transpose2d(out, torch.sum(BC_w * self.EW4, dim=0),
                                                             torch.sum(BC_b * self.EB4, dim=0), 2, 1)))

        out = self.sigmoid(torch.nn.functional.conv_transpose2d(out, torch.sum(BC_w * self.EW5, dim=0),
                                                                torch.sum(BC_b * self.EB5, dim=0), 2, 1))

        return out.view(-1, self.imsize * self.imsize * self.nc)

class ID_Layers(nn.Module):
    def __init__(self, n_e_w=10):
        super(ID_Layers, self).__init__()
        self.n_e_w = n_e_w
        self.loc_layer = nn.Linear(self.n_e_w, self.n_e_w)
        self.scale_layer = nn.Linear(self.n_e_w, self.n_e_w)
        self.softplus = nn.Softplus()
        self.act = nn.LeakyReLU()

    def forward(self, loc, scale):
        loc = self.act(self.loc_layer(loc))
        scale = self.softplus(self.scale_layer(scale))
        return loc, scale

class RNN_encoder(nn.Module):
    def __init__(self, input_dim=10, rnn_dim=10, output_dim=20, n_layers=3, dropout=0.0):
        super(RNN_encoder, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.rnn_dim = rnn_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_dim, rnn_dim, n_layers, bidirectional=True, dropout=dropout)
        self.fc_out = nn.Sequential(nn.Linear(rnn_dim * 2, output_dim), nn.Dropout(dropout))
        self.act = nn.LeakyReLU()

    def forward(self, inp, h, c):
        output_raw, (h, c) = self.rnn(inp, (h, c))
        output = self.act(self.fc_out(output_raw))
        output_forward_raw = output_raw[:, :, :self.rnn_dim]
        return output, h, c, output_forward_raw, output_raw


class GatedTransition(nn.Module):

    def __init__(self, z_dim, transition_dim, dynamics_dim):
        super().__init__()
        # initialize the six linear transformations used in the neural network
        self.lin_gate_z_to_hidden = nn.Linear(z_dim + dynamics_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim + dynamics_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim + dynamics_dim, z_dim)
        # modify the default initialization of lin_z_to_loc
        # so that it's starts out as the identity function
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)
        # initialize the three non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

        self.weight_init()

    def weight_init(self):
        kaiming_init(self.lin_gate_z_to_hidden)
        kaiming_init(self.lin_gate_hidden_to_z)
        kaiming_init(self.lin_proposed_mean_z_to_hidden)
        kaiming_init(self.lin_proposed_mean_hidden_to_z)
        kaiming_init(self.lin_sig)
        kaiming_init(self.lin_z_to_loc)


    def forward(self, z_t_1, dz):
        z_t_1_cat = torch.cat((z_t_1, dz), -1)
        # compute the gating function
        _gate = self.relu(self.lin_gate_z_to_hidden(z_t_1_cat))
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))
        # compute the 'proposed mean'
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1_cat))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)
        # assemble the actual mean used to sample z_t, which mixes a linear transformation
        # of z_{t-1} with the proposed mean modulated by the gating function
        loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        # compute the scale used to sample z_t, using the proposed mean from
        # above as input the softplus ensures that scale is positive
        scale = self.softplus(self.lin_sig(self.relu(proposed_mean)))
        # return loc, scale which can be fed into Normal
        return loc, scale
#

class Combiner(nn.Module):
    def __init__(self, z_dim, rnn_dim, dynamics_dim):
        super().__init__()
        # initialize the three linear transformations used in the neural network
        self.rnn_dim = rnn_dim
        self.h_rnn_to_hidden = nn.Linear(dynamics_dim + rnn_dim, rnn_dim)
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(self.rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(self.rnn_dim, z_dim)
        # initialize the two non-linearities used in the neural network
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.act = nn.LeakyReLU()

    def forward(self, z_t_1, h_rnn, dz):

        h_rnn = torch.cat((h_rnn, dz), -1)
        h_rnn = self.act(self.h_rnn_to_hidden(h_rnn))
        # combine the rnn hidden state with a transformed version of z_t_1
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        # use the combined hidden state to compute the mean used to sample z_t
        loc = self.lin_hidden_to_loc(h_combined)
        # use the combined hidden state to compute the scale used to sample z_t
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        # return loc, scale which can be fed into Normal
        return loc, scale

#
# class Combiner(nn.Module):
#     def __init__(self, z_dim, rnn_dim, dynamics_dim):
#         super().__init__()
#         # initialize the three linear transformations used in the neural network
#         self.rnn_dim = rnn_dim
#         self.lin_hidden_to_loc = nn.Linear(self.rnn_dim + z_dim, z_dim)
#         self.lin_hidden_to_scale = nn.Linear(self.rnn_dim + z_dim, z_dim)
#         # initialize the two non-linearities used in the neural network
#         self.tanh = nn.Tanh()
#         self.softplus = nn.Softplus()
#         self.act = nn.LeakyReLU()
#
#     def forward(self, z_t_1, h_rnn, dz):
#         h_combined = torch.cat((z_t_1, h_rnn), -1)
#         loc = self.lin_hidden_to_loc(h_combined)
#         # use the combined hidden state to compute the scale used to sample z_t
#         scale = self.softplus(self.lin_hidden_to_scale(h_combined))
#         # return loc, scale which can be fed into Normal
#         return loc, scale
