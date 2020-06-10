from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np


import sys
sys.path.append('../')
from Encoder.models.resnet import ResNet18
from config import get_args
from utils.utils import progress_bar
from utils.makedataset import make_cnn_dataset
from utils.dataset import EncoderData

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
        self.bn1 = nn.BatchNorm1d(hidden_units)
        self.linear2 = nn.Linear(hidden_units, hidden_units)
        self.bn2 = nn.BatchNorm1d(hidden_units)
        self.linear3 = nn.Linear(hidden_units, hidden_units)
        self.bn3 = nn.BatchNorm1d(hidden_units)
        self.linear4 = nn.Linear(hidden_units, hidden_units)
        self.bn4 = nn.BatchNorm1d(hidden_units)
        self.linear5 = nn.Linear(hidden_units, 2)  # binary classification, connect or not
    
    def forward(self, x, softmax=False, dropout=0.2):
        out = self.linear1(x)
        out = self.bn1(out)
        out = F.relu(out)
        # out = F.dropout(out, dropout)
        out = self.linear2(out)
        out = self.bn2(out)
        out = F.relu(out)
        # out = F.dropout(out, dropout)
        out = self.linear3(out)
        out = self.bn3(out)
        out = F.relu(out)
        # out = F.dropout(out, dropout)
        out = self.linear4(out)
        out = self.bn4(out)
        out = F.relu(out)
        # out = F.dropout(out, dropout)
        out = self.linear5(out)
        if softmax:
            out = F.softmax(out, dim=-1)
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
        param args.cnn_backbone: The CNN backbone
        param args.hidden_units: hidden_units in ConnetClassify, default is 512
        param args.gpu: The avaliable gpu ids, should be list if gpu is avaliable
        '''
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cnn_backbone = args.cnn_backbone
        self.cnn = makeCNN(self.cnn_backbone)
        self.cnn = self.cnn.to(self.device)
        self.connect = ConnetClassify(self.cnn.feature_size, args.hidden_units).to(self.device)
        self.args = args
        if args.finetune:
            # load the ImageNet-pretrained weights
            self.initialize()
        
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
        return p_connect: the output of the connection (no softmax)
        return sim: the similarity(L2 distance) of two observation
        '''
        pre_input = pre_input.to(self.device)
        now_input = now_input.to(self.device)

        pre_obs = self.cnn(pre_input)
        now_obs = self.cnn(now_input)
        p_connect = self.connect(torch.cat([pre_obs, now_obs],dim=-1))
        # sim = F.cosine_similarity(pre_obs, now_obs)
        sim = torch.norm(pre_obs-now_obs,dim=-1)

        return pre_obs, now_obs, p_connect, sim

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

        now_obs = self.cnn(now_input)
        p_connect = self.connect(torch.cat([observ, now_obs], dim=-1), softmax=True)
        
        return now_obs, p_connect

    def getobservefromimg(self, img):
        '''
        Get the observe from the input image
        param img: the input image which should have the size: 3xwidthxhegiht, and it should be torch.tensor
        return feature: the observe extracted by the CNN
        '''
        # img = torch.Tensor(img).float()
        now_input = img.to(self.device)
        now_obs = self.cnn(now_input)
        return now_obs

    def predictconnect(self, obs1, obs2):
        '''
        Predict the probability of the connection
        :param obs1: the first observation, should be numpy.array
        :param obs2: the second observation, should be numpy.array
        return p_connect: the probability of the connection
        '''
        # obs1 = Variable(torch.from_numpy(obs1))
        # obs1 = Variable(torch.from_numpy(obs2))
        obs1 = torch.from_numpy(obs1).to(self.device)
        obs2 = torch.from_numpy(obs2).to(self.device)
        p_connect = self.connect(torch.cat([obs1, obs2], dim=-1), softmax=True)
        return torch.squeeze(p_connect).cpu().detach().numpy()


    def save_model(self, path):
        '''
        Save the models
        param path: the path of the folder to save the models
        '''
        if not os.path.exists(path):
            os.makedirs(path)
        print("Saving weights in "+path)
        torch.save(self.cnn.state_dict(), os.path.join(path, 'cnn_maxD{}_disR{}.pt'.format(self.args.max_node_dis, self.args.node_dis_reg)))
        torch.save(self.connect.state_dict(), os.path.join(path, 'connect_maxD{}_disR{}.pt'.format(self.args.max_node_dis, self.args.node_dis_reg)))

    def load_model(self, path):
        '''
        Load the models from the folder
        param path: the path of the folder which the weights are saved
        '''
        assert os.path.isfile(os.path.join(
            path, 'cnn_maxD{}_disR{}.pt'.format(self.args.max_node_dis, self.args.node_dis_reg))), '{} is not existed, please check the path'.format(os.path.join(path, 'cnn_maxD{}_disR{}.pt'.format(self.args.max_node_dis, self.args.node_dis_reg)))
        assert os.path.isfile(os.path.join(
            path, 'connect_maxD{}_disR{}.pt'.format(self.args.max_node_dis, self.args.node_dis_reg))), '{} is not existed, please check the path'.format(os.path.join(path, 'connect_maxD{}_disR{}.pt'.format(self.args.max_node_dis, self.args.node_dis_reg)))
        
        print("Loading weights from "+ path)
        self.cnn.load_state_dict(torch.load(os.path.join(path, 'cnn_maxD{}_disR{}.pt'.format(self.args.max_node_dis, self.args.node_dis_reg)), map_location=lambda storage, loc: storage))
        self.connect.load_state_dict(torch.load(os.path.join(path, 'connect_maxD{}_disR{}.pt'.format(self.args.max_node_dis, self.args.node_dis_reg)), map_location=lambda storage, loc: storage))
    
    def initialize(self):
        '''
        Initialize the cnn networks from pretrained model in ImageNet
        '''
        print('Loading the pretrained model from Imagenet')
        if self.cnn_backbone == 'resnet18':
            model = torchvision.models.resnet18(pretrained=True)
        elif self.cnn_backbone == 'resnet34':
            model = torchvision.models.resnet34(pretrained=True)
        elif self.cnn_backbone == 'resnet50':
            model = torchvision.models.resnet50(pretrained=True)
        elif self.cnn_backbone == 'resnet101':
            model = torchvision.models.resnet101(pretrained=True)
        elif self.cnn_backbone == 'resnet152':
            model = torchvision.models.resnet152(pretrained=True)
        else:
            raise NotImplementedError
        pretrained_dict = model.state_dict()
        model_dict = self.cnn.state_dict()
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.cnn.load_state_dict(model_dict)


def trainEncoder(args, encoder, trainloader, testloader, device):
    criterion = nn.CrossEntropyLoss()
    mse_crit = nn.MSELoss()

    cnn_optimizer = optim.Adam(encoder.cnn.parameters(), lr=args.encoder_lr*0.1)
    connect_optimizer = optim.Adam(encoder.connect.parameters(), lr=args.encoder_lr)
    cnn_optimizer.zero_grad()
    connect_optimizer.zero_grad()

    encoder.cnn.train()
    encoder.connect.train()
    best_accuracy = 0
    for epoch in range(args.encoder_epoch):
        print('\nEpoch: %d' % epoch)
        c_train_loss = 0
        cnn_train_loss = 0
        correct = 0

        correct_m1 = 0
        correct_1 = 0
        correct_0 = 0

        total = 0
        encoder.cnn.train()
        encoder.connect.train()
        for batch_idx, (img1, img2, targets) in enumerate(trainloader):
            # targets: 1 connect, 0 not connect (two nodes), -1 not connect (one node)
            targets = targets.to(device)
            _, _, outputs, sim = encoder.predconnectfromimg(img1, img2)

            connect_targets = targets.clone()
            connect_targets[targets==-1] = 0
            c_loss_pred = criterion(outputs, connect_targets)

            _, predicted = outputs.max(1)
            outputs_soft = F.softmax(outputs,dim=-1)

            if sim[targets==-1].shape == torch.Size([0]):
                c_loss = c_loss_pred.clone()
                c_loss_cnn = c_loss_pred.clone()
            else:
                c_loss_cnn = mse_crit(sim[targets == -1], torch.zeros_like(sim[targets == -1]).to(device)) * args.node_dis_reg + c_loss_pred
                c_loss = c_loss_pred
            
            if sim[targets==1].shape != torch.Size([0]):
                c_loss_cnn = c_loss_cnn + mse_crit(sim[targets == 1], torch.ones_like(sim[targets == 1]).to(device) * args.max_node_dis) * args.node_dis_reg

            c_loss.backward(retain_graph=True)
            connect_optimizer.step()
            connect_optimizer.zero_grad()

            cnn_optimizer.zero_grad()

            c_loss_cnn.backward()
            cnn_optimizer.step()
            cnn_optimizer.zero_grad()

            connect_optimizer.zero_grad()

            
            cnn_train_loss += c_loss_cnn.item()
            c_train_loss += c_loss.item()
            
            total += targets.size(0)
            correct += predicted.eq(connect_targets).sum().item()

            correct_m1 += predicted[targets == -1].eq(connect_targets[targets == -1]).sum().item()
            correct_0 += predicted[targets == 0].eq(connect_targets[targets == 0]).sum().item()
            correct_1 += predicted[targets == 1].eq(connect_targets[targets == 1]).sum().item()

            progress_bar(batch_idx, len(trainloader), 'connect loss: %.3f | cnn loss: %.3f | Acc: %.3f%% (%d/%d) |inc -1:%.2f%% |c: %.2f%% |inc:%.2f%%'
                         % (c_train_loss/(batch_idx+1), cnn_train_loss/(batch_idx+1), 100.*correct/total, correct, total, 100.*correct_m1/total, 100.*correct_1/total, 100.*correct_0/total))
        
        test_loss = 0
        test_correct = 0
        test_correct_m1 = 0
        test_correct_0 = 0
        test_correct_1 = 0
        test_total = 0
        encoder.cnn.eval()
        encoder.connect.eval()

        for batch_idx, (img1, img2, targets) in enumerate(testloader):
            targets = targets.to(device)
            
            with torch.no_grad():
                _,_,outputs, _ = encoder.predconnectfromimg(img1, img2)
            cnn_targets = targets.clone()
            cnn_targets[targets==-1] = 0
            loss = criterion(outputs, cnn_targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(cnn_targets).sum().item()

            test_correct_m1 += predicted[targets == -1].eq(cnn_targets[targets == -1]).sum().item()
            test_correct_0 += predicted[targets == 0].eq(cnn_targets[targets == 0]).sum().item()
            test_correct_1 += predicted[targets == 1].eq(cnn_targets[targets == 1]).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) |inc -1:%.2f%% |c: %.2f%% |inc:%.2f%%'
                         % (test_loss/(batch_idx+1), 100.*test_correct/test_total, test_correct, test_total, 100.*test_correct_m1/test_total, 100.*test_correct_1/test_total, 100.*test_correct_0/test_total))

        if best_accuracy < (test_correct/test_total):
            best_accuracy = (test_correct/test_total)
            encoder.save_model(os.path.join(args.out_dir, 'min_{}_max_{}'.format(args.connect_min, args.connect_max), 'encoder_new', 'best'))
        else:
            encoder.save_model(os.path.join(args.out_dir, 'min_{}_max_{}'.format(args.connect_min, args.connect_max), 'encoder_new', 'latest'))
    
    return encoder


def testEncoder(args, encoder, testloader, device):
    test_loss = 0
    test_correct = 0
    test_total = 0
    test_correct_m1 = 0
    test_correct_0 = 0
    test_correct_1 = 0
    criterion = nn.CrossEntropyLoss()
    encoder.cnn.eval()
    encoder.connect.eval()
    for batch_idx, (img1, img2, targets) in enumerate(testloader):
        targets = targets.to(device)
        with torch.no_grad():
            _,_,outputs,_ = encoder.predconnectfromimg(img1, img2)
        cnn_targets = targets.clone()
        cnn_targets[targets==-1] = 0
        loss = criterion(outputs, cnn_targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        test_total += targets.size(0)
        test_correct += predicted.eq(cnn_targets).sum().item()

        test_correct_m1 += predicted[targets == -1].eq(cnn_targets[targets == -1]).sum().item()
        test_correct_0 += predicted[targets == 0].eq(cnn_targets[targets == 0]).sum().item()
        test_correct_1 += predicted[targets == 1].eq(cnn_targets[targets == 1]).sum().item()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*test_correct/test_total, test_correct, test_total))
    print("same node:{}%; inconnect node:{}%; connect node:{}%".format(100.*test_correct_m1/test_total, 100.*test_correct_0/test_total, 100.*test_correct_1/test_total))

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(args)
    if not os.path.isfile(os.path.join(args.path_train_test, 'min_{}_max_{}'.format(args.connect_min, args.connect_max), 'training.txt')):
        print('The training set and testing set are not divided. Make them now')
        make_cnn_dataset(args.path_images, args.path_train_test, conn_max=args.connect_max, conn_min=args.connect_min, data_num=args.img_dataset_num)
    
    trainloader = DataLoader(EncoderData(args.path_images, os.path.join(args.path_train_test, 'min_{}_max_{}'.format(args.connect_min, args.connect_max), 'training.txt'), 
                            target_transform=transforms.Compose([transforms.Resize(args.img_width),
                                                                transforms.ToTensor()])),
                            batch_size=args.encoder_batchsize, shuffle=True, num_workers=4)

    testloader = DataLoader(EncoderData(args.path_images, os.path.join(args.path_train_test, 'min_{}_max_{}'.format(args.connect_min, args.connect_max), 'testing.txt'), 
                            target_transform=transforms.Compose([transforms.Resize(args.img_width),
                                                                transforms.ToTensor()])),
                            batch_size=args.encoder_batchsize, shuffle=False, num_workers=4)

    if args.test_encoder:
        encoder.load_model(os.path.join(args.out_dir, 'min_{}_max_{}'.format(args.connect_min, args.connect_max), 'encoder_new', 'best'))
        testEncoder(args, encoder, testloader, device)
    else:
        if args.load_encoder:
            encoder.load_model(os.path.join(args.out_dir, 'min_{}_max_{}'.format(args.connect_min, args.connect_max), 'encoder_new', 'best'))
        trainEncoder(args, encoder, trainloader, testloader, device)
    



if __name__ == '__main__':
    args = get_args()
    gpus = ''
    for ids in args.gpu:
        gpus+=str(ids)
        gpus+=','
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    
    main(args)
