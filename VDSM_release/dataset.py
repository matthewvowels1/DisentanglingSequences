

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import nonechucks as nc
import random
from sklearn.model_selection import train_test_split


class MyMUGDataset(Dataset):
    def __init__(self, data, imsize, slice_length, flip_flag):
        self.data = data
        self.imsize = imsize
        self.slice_length = slice_length
        self.flip_flag = flip_flag

    def _transform(self):
        options = []
        options.append(transforms.ToPILImage())
        options.append(transforms.CenterCrop(160))
        options.append(transforms.Resize((64, 64)))
        options.append(transforms.ToTensor())
        if self.flip_flag:
            options.append(transforms.RandomHorizontalFlip(0.5))
        transform = transforms.Compose(options)
        return transform

    def __getitem__(self, index):
        transform = self._transform()
        seq = self.data[0][index][::2]
        labels = self.data[1][index]
        if labels[1] != 6 and labels[1] != 5 and labels[1] != 2:

            if labels[1] == 8:  # reassign labels
                labels[1] = 5
            elif labels[1] == 7:
                labels[1] = 2

            start = random.randint(0, len(seq) - self.slice_length)
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


def create_MUG_dataset(folder, imsize, slice_length, seed, pretrain):
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load_ = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    npz_file = os.path.join(folder, 'MUG_0-255_OF_112x112_color.npz')

    file_labels = os.path.join(folder, 'labels.npz')

    labels = np.load(file_labels)['arr_0']
    data = np.load_(npz_file)['arr_0']
    imgs_train, imgs_test, labels_train, labels_test = train_test_split(data, labels, test_size=.25, random_state=seed)
    data_train = [imgs_train, labels_train]
    data_test = [imgs_test, labels_test]

    flip_flag = True if pretrain else False
    dataset_train = nc.SafeDataset(MyMUGDataset(data_train, imsize, slice_length, flip_flag))
    dataset_test = nc.SafeDataset(MyMUGDataset(data_test, imsize, slice_length, flip_flag))
    print('data train shape', imgs_train.shape)
    print('data test shape', imgs_test.shape)
    return dataset_train, dataset_test

def create_sprites_dataset(path, num_train, num_test):
    dataset_train = MySpritesDataset(os.path.join(path, 'train'), num_train)
    dataset_test = MySpritesDataset(os.path.join(path, 'test'), num_test)
    return dataset_train, dataset_test


class MySpritesDataset(Dataset):
    # https://github.com/yatindandi/Disentangled-Sequential-Autoencoder/blob/master/trainer.py
    def __init__(self, path, size):
        self.path = path
        self.length = size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rand_action = np.random.randint(9)
        id_ = idx // 9  # there are 9 actions so this gets us into a single id
        first_action_index = (id_ * 9) + rand_action

        rand_action = np.random.randint(9)
        second_action_index = (id_ * 9) + rand_action

        first_item = torch.load(self.path + '/%d.sprite' % (first_action_index + 1))
        second_item = torch.load(self.path + '/%d.sprite' % (second_action_index + 1))
        #         item = torch.load(self.path + '/%d.sprite' % (0 + 1))

        first_seq = (first_item['sprite'] + 1) / 2
        second_seq = (second_item['sprite'] + 1) / 2
        x = torch.cat((first_seq, second_seq), 0)

        return x, first_item

