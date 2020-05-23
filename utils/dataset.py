from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import sys
from PIL import Image
import random
from utils.makedataset import get_images_list

class EncoderData(Dataset):
    '''
    The dataloader for training Encoder
    '''
    def __init__(self, img_path, name_path, target_transform = None):
        '''
        :param img_path: the path to the raw images
        :param name_path: the path to the txt file
        '''
        self.img_path = img_path
        self.data = self.read_txt(name_path)
        self.transform = target_transform
        
    def read_txt(self, path):
        data = []
        with open(path, 'r') as f:
            for line in f.readlines():
                line = line.strip().split(' ')
                data.append([line[0], line[1], int(line[2])])
        return data

    def __getitem__(self, index):
        [img1_name, img2_name, target] = self.data[index]
        names = [img1_name, img2_name]
        random.shuffle(names)
        img1 = Image.open(os.path.join(self.img_path, names[0]))
        img2 = Image.open(os.path.join(self.img_path, names[1]))
        if self.transform is not None:
            img1 = self.transform(img1) 
            img2 = self.transform(img2) 
        return img1, img2, target

    def __len__(self):
        return len(self.data)

class SimulateConstructGraphData(Dataset):
    '''
    The dataloader for contrusting the graph in simulation
    '''
    def __init__(self, img_path, target_transform = None):
        '''
        :param img_path: the path to the raw images
        :param name_path: the path to the txt file
        '''
        self.img_path = img_path
        self.names = sorted(get_images_list(self.img_path))

        self.transform = target_transform

    def __getitem__(self, index):
        img_name = self.names[index]
        img = Image.open(os.path.join(self.img_path, img_name))
        heading = img_name.split('_')[1]
        coord_x = float(img_name.split('_')[2])
        coord_y = float(img_name.split('_')[3])
        
        if self.transform is not None:
            img = self.transform(img) 
        return img, [coord_x, coord_y, heading]

    def __len__(self):
        return len(self.names)