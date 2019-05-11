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

class MnistdfTestDataset(BaseDataset):
    def name(self):
        return 'MnistdfTestDataset'

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        print(opt)
	x_adv_train, x_adv_test = pickle.load(open(os.path.join(opt.dataroot, "df/df.pkl")))
	x_train, x_test, y_train, y_test = pickle.load(open(os.path.join(opt.dataroot, "mnist/mnist.pkl")))

        self.mnist = x_test
	self.mnist = resize(self.mnist);
        self.mnist_label = y_test

        self.mnistdf = np.concatenate((x_adv_test, x_test))
	self.mnistdf = resize(self.mnistdf)
        self.mnistdf_label = np.concatenate((y_test, y_test))
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])

        self.shuffle_indices()

    def shuffle_indices(self):
        self.mnist_indices = list(range(self.mnist.shape[0]))
        self.mnistdf_indices = list(range(self.mnistdf.shape[0]))
        print('num mnist', len(self.mnist_indices), 'num mnistdf', len(self.mnistdf_indices))
        if not self.opt.serial_batches:
            random.shuffle(self.mnist_indices)
            random.shuffle(self.mnistdf_indices)

    def __getitem__(self, index):

        if index == 0:
            self.shuffle_indices()

        A_img = self.mnist[self.mnist_indices[index % self.mnist.shape[0]]]
	A_label = self.mnist_label[self.mnist_indices[index % self.mnist.shape[0]]]
        A_img = self.transform(A_img)
        A_path = '%01d_%05d.png' % (A_label, index)

        B_img = self.mnistdf[self.mnistdf_indices[index % self.mnistdf.shape[0]]]
        B_label = self.mnistdf_label[self.mnistdf_indices[index % self.mnistdf.shape[0]]]
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

        return self.mnistdf.shape[0] #min(len(self.mnist), self.svhn.shape[0])
        
