import torch.nn as nn
import torch.nn.init as init
import os
from natsort import natsorted


def check_for_checkpt(folder, suffix):
    files = os.listdir(folder)
    model_pths = []
    for file in files:
        if suffix in file:
            model_pths.append(file)
    if len(model_pths) != 0:
        print('pre-existing checkpoint found')
        sorted_pths = natsorted(model_pths)
        most_recent = os.path.join(folder, sorted_pths[-1])
        print('loading model from: ',most_recent )
        epoch = int(sorted_pths[-1].split('_')[0])
    else:
        epoch = 0
        most_recent = None
    return most_recent, epoch

def make_folder(path, version):
        if not os.path.exists(os.path.join(path, version)):
            os.makedirs(os.path.join(path, version))

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

