import argparse


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')

    # The configure of Encoder
    parser.add_argument('--img_width', type=int, default=256,
                        help='the width size of image (default: 768)')
    parser.add_argument('--img_height', type=int, default=256,
                        help='the height size of image (default: 768)')
    parser.add_argument('--cnn_backbone', type=str, default='resnet18')
    parser.add_argument('--hidden_units', type=int, default=512,
                        help='the hidden units in the FC layers in connectclassify (default: 512)')
    parser.add_argument('--encoder_batchsize', type=int, default=10, metavar='N',
                        help='batch size of the encoder during training (default: 32)')
    parser.add_argument('--encoder_epoch', type=int, default=100, metavar='N',
                        help='number of epochs of the encoder during training (default: 100)')
    parser.add_argument('--encoder_lr', type=float, default=0.001,
                        help='the learning rate of the encoder')
    parser.add_argument('--connect_min', type=float, default=2.0,
                        help='the minimum distance to determine whether two images are connected')
    parser.add_argument('--connect_max', type=float, default=3.0,
                        help='the maximum distance to determine whether two images are connected')
    parser.add_argument('--path_images', type=str, default='../dataset/carla_rawdata/', 
                        help='the path to the saved images')
    parser.add_argument('--path_train_test', type=str, default='../dataset/', 
                        help='the path to the files which stored the train set and test set image names')
    parser.add_argument('--load_encoder', action='store_true',
                        help='load the encoder weights')
    parser.add_argument('--test_encoder', action='store_true',
                        help='whether test the encoder')
    parser.add_argument('--finetune', action='store_true', default=True,
                        help='load the imagenet pretrained weights')
    parser.add_argument('--connect_threshold', type=float, default=0.93,
                        help='the threshold to determine whether the two observations are connected')
    parser.add_argument('--constructgraph_topK', type=int, default=2,
                        help='add the top K edges when contructing the new graph')


    parser.add_argument('--gpu', default=[1], help='the avaliable gpu ids')
    parser.add_argument('--out_dir', type=str, default='./wights', help='the dir to save the weights')
    args = parser.parse_args()
    return args
