import random
import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
import scipy.io
import numpy as np
import pickle
import cv2

from PIL import Image
from PIL.ImageOps import invert

class Cifar10Cifar10fgDataset(BaseDataset):
    def name(self):
        return 'Cifar10Cifar10fgDataset'

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        print(opt)
	x_adv_train, x_adv_test = pickle.load(open(os.path.join(opt.dataroot, "fg/cifar_fg.pkl")))
	x_train, x_test, y_train, y_test = pickle.load(open(os.path.join(opt.dataroot, "cifar10/cifar10.pkl")))
	y_train = y_train.reshape([-1]).astype('int64')
	y_test = y_test.reshape([-1]).astype('int64')

        self.cifar10 = x_train
        self.cifar10_label = y_train

        self.cifar10fg = np.concatenate((x_adv_train, x_train))
        self.cifar10fg_label = np.concatenate((y_train, y_train))
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])

        self.shuffle_indices()

    def shuffle_indices(self):
        self.cifar10_indices = list(range(self.cifar10.shape[0]))
        self.cifar10fg_indices = list(range(self.cifar10fg.shape[0]))
        print('num cifar10', len(self.cifar10_indices), 'num cifar10fg', len(self.cifar10fg_indices))
        if not self.opt.serial_batches:
            random.shuffle(self.cifar10_indices)
            random.shuffle(self.cifar10fg_indices)

    def __getitem__(self, index):

        if index == 0:
            self.shuffle_indices()

        A_img = self.cifar10[self.cifar10_indices[index % self.cifar10.shape[0]]]
	A_label = self.cifar10_label[self.cifar10_indices[index % self.cifar10.shape[0]]]
        A_img = self.transform(A_img)
        A_path = '%01d_%05d.png' % (A_label, index)

        B_img = self.cifar10fg[self.cifar10fg_indices[index % self.cifar10fg.shape[0]]]
        B_label = self.cifar10fg_label[self.cifar10fg_indices[index % self.cifar10fg.shape[0]]]
        B_img = self.transform(B_img)
        B_path = '%01d_%05d.png' % (B_label, index)

        item = {}
        item.update({'A': A_img,
                     'A_paths': A_path,
                     'A_label': A_label
                 })
        
        item.update({'B': B_img,
                     'B_paths': B_path,
                     'B_label': B_label
                 })
        return item
        
    def __len__(self):
        #if self.opt.which_direction == 'AtoB':
        #    return len(self.cifar10)
        #else:            
        #    return self.svhn.shape[0]

        return self.cifar10fg.shape[0] #min(len(self.cifar10), self.svhn.shape[0])
        
