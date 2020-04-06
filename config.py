import argparse


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')

    # The configure of Encoder
    parser.add_argument('--img_size', type=int, default=84,
                        help='the size of image (default: 84)')
    parser.add_argument('--cnn_backbone', type=str, default='resnet18')
    parser.add_argument('--hidden_units', type=int, default=512,
                        help='the hidden units in the FC layers in connectclassify (default: 512)')
    parser.add_argument('--encoder_batchsize', type=int, default=64, metavar='N',
                        help='batch size of the encoder during training (default: 32)')
    parser.add_argument('--encoder_epoch', type=int, default=1000, metavar='N',
                        help='number of epochs of the encoder during training (default: 100)')
    parser.add_argument('--encoder_lr', type=float, default=0.001,
                        help='the learning rate of the encoder')

    parser.add_argument('--gpu', default=[0], help='the avaliable gpu ids')
    parser.add_argument('--out_dir', type=str, default='./GAM', help='the dir to save the weights')
    args = parser.parse_args()
    return args
