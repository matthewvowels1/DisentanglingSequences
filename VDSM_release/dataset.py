

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import nonechucks as nc
import random
from sklearn.model_selection import train_test_split

class MyMUGDataset(Dataset):
    def __init__(self, data, imsize, slice_length):
        self.data = data
        self.imsize = imsize
        self.slice_length = slice_length

    def _transform(self):
        options = []
        options.append(transforms.ToPILImage())
        options.append(transforms.CenterCrop(160))
        options.append(transforms.Resize((64, 64)))
        options.append(transforms.ToTensor())  # note there is no horizontal flip in here (unlike the i.i.d. case)
        transform = transforms.Compose(options)
        return transform

    def __getitem__(self, index):
        transform = self._transform()
        seq = self.data[0][index][::2]
        labels = self.data[1][index]
        if labels[1] != 6 and labels[1] != 5 and labels[1] != 2:  # ignore neutral, mixed, and extra actions
            start = random.randint(0, len(seq) - self.slice_length)

            if labels[1] == 8:   # reassign labels
                labels[1] = 5
            elif labels[1] == 7:
                labels[1] = 2
            end = start + self.slice_length
            seq_slice = seq[start:end]
            seqs = []
            for im in seq_slice:
                seqs.append(transform(im))
            seqs = torch.stack(seqs)

            return seqs, labels
        else:
            return None

    def __len__(self):
        return len(self.data[0])

def create_MUG_dataset(folder, imsize, slice_length, seed):
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load_ = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    npz_file = os.path.join(folder, 'MUG_0-255_OF_112x112_color.npz')

    file_labels = os.path.join(folder, 'labels.npz')

    labels = np.load(file_labels)['arr_0']
    data = np.load_(npz_file)['arr_0']
    imgs_train, imgs_test, labels_train, labels_test = train_test_split(data, labels, test_size=.2, random_state=seed)
    data_train = [imgs_train, labels_train]
    data_test = [imgs_test, labels_test]

    dataset_train = nc.SafeDataset(MyMUGDataset(data_train, imsize, slice_length))
    dataset_test = nc.SafeDataset(MyMUGDataset(data_test, imsize, slice_length))
    print('data train shape', imgs_train.shape)
    print('data test shape', imgs_test.shape)
    return dataset_train, dataset_test
