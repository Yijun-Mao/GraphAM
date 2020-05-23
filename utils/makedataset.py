import os
import sys
import cv2
import numpy as np
import math
import random

def get_images_list(path):
    '''
    Get all names of the images in current path
    :param path: the path of images
    '''
    names = []
    for i in os.listdir(path):
        if i.endswith('.png'):
            names.append(i)
    return sorted(names)

def compute_dis(img1, img2):
    '''
    Compute the distance between img1 and img2
    '''
    img1 = img1.split('.')[0]
    img2 = img2.split('.')[0]
    coord_x1 = float(img1.split('_')[2])
    coord_y1 = float(img1.split('_')[3])

    coord_x2 = float(img2.split('_')[2])
    coord_y2 = float(img2.split('_')[3])

    return math.sqrt((coord_x1-coord_x2)**2 + (coord_y1-coord_y2)**2)

def train_test_split(all_files, ratio):
    random.shuffle(all_files)
    connect_train, connect_test = all_files[0:int(len(all_files)*ratio)], all_files[int(len(all_files)*ratio):]
    return connect_train, connect_test

def make_cnn_dataset(path_raw, path_save, conn_min, conn_max, data_num=10000, ratio=0.8, connect_ratio=0.4):
    '''
    Make the dataset from carla_raw_data used for training the CNN in Encoder.
    1 represents the two images are connected, 0 represents the two images are not connected.
    :param path_raw: the path of the raw data
    :param path_save: the path to save the txt files which contain the names of two images
    :param conn_min: the minimum distance to determine whether two images are connected
    :param conn_max: the maximum distance to determine whether two images are connected
    :param data_num: the total number of the pairs made by the raw data
    :param ratio: the ratio of training set and testing set
    :param connect_ratio: the ratio of the connect pairs in all pairs
    '''
    imgs = get_images_list(path_raw)
    total_imgs = len(imgs)
    print('There are {} raw images in total.'.format(total_imgs))
    connect = set()
    inconnect2node = set()
    inconnect1node = set()
    i=0
    while len(connect) < int(data_num*connect_ratio):
        i=i%total_imgs
        idx = random.randint(-25, 25)
        distance = compute_dis(imgs[i], imgs[(i+idx+total_imgs)%total_imgs])
        if distance < conn_max and distance > conn_min:
            connect.add(' '.join(sorted([imgs[i], imgs[(i+idx+total_imgs)%total_imgs]])))
        i+=1
        if i%2000 == 0:
            print("Get {}/{} images".format(len(connect), int(data_num*connect_ratio)))
    print("connect done")
    i=0
    while len(inconnect2node) < (data_num - len(connect))//2:
        i = i % total_imgs
        idx = random.randint(0, total_imgs)
        distance = compute_dis(imgs[i], imgs[(i+idx+total_imgs)%total_imgs])
        if distance >= conn_max:# or distance <= conn_min:
            inconnect2node.add(' '.join(sorted([imgs[i], imgs[(i+idx+total_imgs)%total_imgs]])))
        i+=1
        if i%2000 == 0:
            print("Get {}/{} images".format(len(inconnect2node), (data_num - len(connect))//2))
    print("inconnect2 done")
    i=0
    while len(inconnect1node) < (data_num - len(connect) - len(inconnect2node)):
        i = i % total_imgs
        idx = random.randint(-20, 20)
        distance = compute_dis(imgs[i], imgs[(i+idx+total_imgs)%total_imgs])
        if distance <= conn_min:
            inconnect1node.add(' '.join(sorted([imgs[i], imgs[(i+idx+total_imgs)%total_imgs]])))
        i += 1
        if i%2000 == 0:
            print("Get {}/{} images".format(len(inconnect1node), data_num - len(connect) - len(inconnect2node)))
    print("inconnect1 done")

    connect_train, connect_test = train_test_split(list(connect), ratio)
    inconnect1node_train, inconnect1node_test = train_test_split(list(inconnect1node), ratio)
    inconnect2node_train, inconnect2node_test = train_test_split(list(inconnect2node), ratio)
    
    allpairs_train = []
    for pair in connect_train:
        pair = pair.split(' ')
        allpairs_train.append([pair[0], pair[1], 1])

    for pair in inconnect1node_train:
        pair = pair.split(' ')
        allpairs_train.append([pair[0], pair[1], -1])

    for pair in inconnect2node_train:
        pair = pair.split(' ')
        allpairs_train.append([pair[0], pair[1], 0]) # in the same node
    
    allpairs_test = []
    for pair in connect_test:
        pair = pair.split(' ')
        allpairs_test.append([pair[0], pair[1], 1])

    for pair in inconnect1node_test:
        pair = pair.split(' ')
        allpairs_test.append([pair[0], pair[1], -1])
        
    for pair in inconnect2node_test:
        pair = pair.split(' ')
        allpairs_test.append([pair[0], pair[1], 0]) # in the same node

    random.shuffle(allpairs_train)
    random.shuffle(allpairs_test)

    trainset = allpairs_train
    testset = allpairs_test

    path_save = os.path.join(path_save, 'min_{}_max_{}'.format(conn_min, conn_max))

    if not os.path.exists(path_save):
        os.makedirs(path_save)
    with open(os.path.join(path_save,'training.txt'), 'w') as f:
        for pair in trainset:
            f.write('{0} {1} {2}\n'.format(pair[0], pair[1], pair[2]))

    with open(os.path.join(path_save,'testing.txt'), 'w') as f:
        for pair in testset:
            f.write('{0} {1} {2}\n'.format(pair[0], pair[1], pair[2]))

if __name__ == "__main__":
    make_cnn_dataset('../dataset/carla_rawdata', '../dataset/', conn_max=4.0, conn_min=1.5, data_num=10000)
