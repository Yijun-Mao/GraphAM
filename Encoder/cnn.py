from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np

from models import *
import sys
sys.path.append('../')
from config import get_args
from utils.utils import progress_bar

class ConnetClassify(nn.Module):
    '''
    Predict whether two observation are connected.
    '''
    def __init__(self, feature_size, hidden_units=512):
        '''
        :param feature_size: the size of feature extracted by CNN encoder. The input should be the concatenation of two observations.
        :param hidden_units: the hidden units in the FC layers
        '''
        super(ConnetClassify, self).__init__()
        self.feature_size = feature_size
        self.hidden_units = hidden_units

        self.linear1 = nn.Linear(self.feature_size*2, hidden_units)
        self.linear2 = nn.Linear(hidden_units, hidden_units)
        self.linear3 = nn.Linear(hidden_units, hidden_units)
        self.linear4 = nn.Linear(hidden_units, hidden_units)
        self.linear5 = nn.Linear(hidden_units, 2)  # binary classification, connect or not
    
    def forward(self, x):
        out = self.linear1(x)
        out = F.ReLU(out)
        out = self.linear2(out)
        out = F.ReLU(out)
        out = self.linear3(out)
        out = F.ReLU(out)
        out = self.linear4(out)
        out = F.ReLU(out)
        out = self.linear5(out)
        out = F.softmax(out)
        return out

def makeCNN(cnn_backbone='resnet18'):
    '''
    Initialize cnn
    '''
    if cnn_backbone == 'resnet18':
        cnn = ResNet18(classify=False)
    elif cnn_backbone == 'resnet34':
        cnn = ResNet34(classify=False)
    elif cnn_backbone == 'resnet50':
        cnn = ResNet50(classify=False)
    elif cnn_backbone == 'resnet101':
        cnn = ResNet101(classify=False)
    elif cnn_backbone == 'resnet152':
        cnn = ResNet152(classify=False)
    elif cnn_backbone == 'VGG19':
        cnn = VGG('VGG19', classify=False)
    else:
        raise AssertionError(
            'The backbone is not implemented currently. It should be in [Resnet18, Resnet34, Resnet50, Resnet101, Resnet152, VGG].')
    return cnn

class Encoder(object):
    '''
    The Encoder module, which contains the CNN extractor and connection classification.
    '''
    def __init__(self, args):
        super(Encoder).__init__()
        '''
        param cnn_backbone: The CNN backbone
        param hidden_units: hidden_units in ConnetClassify, default is 512
        param gpu: The avaliable gpu ids, should be list if gpu is avaliable
        '''
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cnn_backbone = args.cnn_backbone
        self.cnn = makeCNN(self.cnn_backbone)
        self.cnn = self.cnn.to(self.device)
        self.connect = ConnetClassify(self.cnn.feature_size, args.hidden_units).to(self.device)
        # if args.finetune:
        #     self.initialize()
        
        if torch.cuda.device_count() > 1:
            self.cnn = nn.DataParallel(self.cnn, device_ids=[int(i) for i in args.gpu])
            self.connect = nn.DataParallel(self.connect, device_ids=[int(i) for i in args.gpu])

    def predconnectfromimg(self, pre_input, now_input):
        '''
        Predict the connection of two images
        :param pre_input: the previous input image
        :param now_input: current input image
        return pre_obs: the observation extracted from previous image
        return now_obs: the observation extracted from current image
        return p_connect: the probability of the connection
        '''
        pre_input = pre_input.to(self.device)
        now_input = now_input.to(self.device)

        pre_obs = self.cnn(pre_input)
        now_obs = self.ccnn(now_input)
        p_connect = self.connect(torch.cat([pre_obs, now_obs],dim=-1))

        return pre_obs, now_obs, p_connect

    def predconnectfromobs(self, observ, img):
        '''
        Predict the connection between previous observation and current input image
        param observ: the previous observation
        param img: the current image
        return now_obs: the current observation
        return p_connect: the probability of the connection
        '''
        now_input = img.to(self.device)
        observ = observ.to(self.device)

        now_obs = self.ccnn(now_input)
        p_connect = self.connect(torch.cat([observ, now_obs], dim=-1))
        
        return now_obs, p_connect

    def save_model(self, path):
        '''
        Save the models
        param path: the path of the folder to save the models
        '''
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.cnn.state_dict(), os.path.join(path, 'cnn.pt'))
        torch.save(self.connect.state_dict(), os.path.join(path, 'connect.pt'))

    def load_model(self, path):
        '''
        Load the models from the folder
        param path: the path of the folder which the weights are saved
        '''
        assert os.path.isfile(os.path.join(
            path, 'cnn.pt')), '{} is not existed, please check the path'.format(os.path.join(path, 'cnn.pt'))
        assert os.path.isfile(os.path.join(
            path, 'connect.pt')), '{} is not existed, please check the path'.format(os.path.join(path, 'connect.pt'))
        
        self.cnn.load_state_dict(torch.load(os.path.join(path, 'cnn.pt')))
        self.connect.load_state_dict(torch.load(os.path.join(path, 'connect.pt')))
    
    def initialize(self):
        '''
        Initialize the cnn networks from pretrained model in ImageNet
        '''
        raise NotImplementedError


def train(args, encoder, trainloader, testloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([{'params': encoder.cnn.parameters()}, {
                           'params': encoder.connect.parameters()}], lr=args.encoder_lr)
    optimizer.zero_grad()
    encoder.cnn.train()
    encoder.connect.train()
    best_accuracy = 0
    for epoch in range(args.encoder_epoch):
        print('\nEpoch: %d' % epoch)
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (img1, img2, targets) in enumerate(trainloader):
            # targets: 0 connect, 1 not connect
            targets = targets.to(device)
            outputs = encoder.predconnectfromimg(img1, img2)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        test_loss = 0
        test_correct = 0
        test_total = 0
        for batch_idx, (img1, img2, targets) in enumerate(testloader):
            targets = targets.to(device)
            outputs = encoder.predconnectfromimg(img1, img2)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*test_correct/test_total, test_correct, test_total))

        if best_accuracy < (test_correct/test_total):
            best_accuracy = (test_correct/test_total)
            encoder.save_model(os.path.join(args.out_dir, 'best'))
        else:
            encoder.save_model(os.path.join(args.out_dir, 'latest'))
    
    return encoder


def test(args, encoder, testloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loss = 0
    test_correct = 0
    test_total = 0
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (img1, img2, targets) in enumerate(testloader):
        targets = targets.to(device)
        outputs = encoder.predconnectfromimg(img1, img2)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        test_total += targets.size(0)
        test_correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*test_correct/test_total, test_correct, test_total))


if __name__ == '__main__':
    args = get_args()
    gpus = ''
    for ids in args.gpu:
        gpus+=str(ids)
        gpus+=','
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus