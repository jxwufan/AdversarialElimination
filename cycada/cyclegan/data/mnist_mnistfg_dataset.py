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

def resize(images):
    output = []
    for i in range(images.shape[0]):
   	output.append(cv2.resize(images[i], (32, 32)))
    output = np.array(output)
    return output

class MnistMnistfgDataset(BaseDataset):
    def name(self):
        return 'MnistMnistfgDataset'

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        print(opt)
	x_adv_train, x_adv_test = pickle.load(open(os.path.join(opt.dataroot, "fg/fg.pkl")))
	x_train, x_test, y_train, y_test = pickle.load(open(os.path.join(opt.dataroot, "mnist/mnist.pkl")))

        self.mnist = x_train
	self.mnist = resize(self.mnist);
        self.mnist_label = y_train

        self.mnistfg = np.concatenate((x_adv_train, x_train))
	self.mnistfg = resize(self.mnistfg)
        self.mnistfg_label = np.concatenate((y_train, y_train))
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])

        self.shuffle_indices()

    def shuffle_indices(self):
        self.mnist_indices = list(range(self.mnist.shape[0]))
        self.mnistfg_indices = list(range(self.mnistfg.shape[0]))
        print('num mnist', len(self.mnist_indices), 'num mnistfg', len(self.mnistfg_indices))
        if not self.opt.serial_batches:
            random.shuffle(self.mnist_indices)
            random.shuffle(self.mnistfg_indices)

    def __getitem__(self, index):

        if index == 0:
            self.shuffle_indices()

        A_img = self.mnist[self.mnist_indices[index % self.mnist.shape[0]]]
	A_label = self.mnist_label[self.mnist_indices[index % self.mnist.shape[0]]]
        A_img = self.transform(A_img)
        A_path = '%01d_%05d.png' % (A_label, index)

        B_img = self.mnistfg[self.mnistfg_indices[index % self.mnistfg.shape[0]]]
        B_label = self.mnistfg_label[self.mnistfg_indices[index % self.mnistfg.shape[0]]]
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
        #    return len(self.mnist)
        #else:            
        #    return self.svhn.shape[0]

        return self.mnistfg.shape[0] #min(len(self.mnist), self.svhn.shape[0])
        
