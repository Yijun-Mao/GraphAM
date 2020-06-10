import argparse


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')

    # The configure of Encoder
    parser.add_argument('--img_width', type=int, default=256,
                        help='the width size of image (default: 256)')
    parser.add_argument('--img_height', type=int, default=256,
                        help='the height size of image (default: 256)')
    parser.add_argument('--cnn_backbone', type=str, default='resnet18')
    parser.add_argument('--hidden_units', type=int, default=512,
                        help='the hidden units in the FC layers in connectclassify (default: 512)')
    parser.add_argument('--encoder_batchsize', type=int, default=80, metavar='N',
                        help='batch size of the encoder during training (default: 80)')
    parser.add_argument('--encoder_epoch', type=int, default=80, metavar='N',
                        help='number of epochs of the encoder during training (default: 80)')
    parser.add_argument('--encoder_lr', type=float, default=0.001,
                        help='the learning rate of the encoder')
    parser.add_argument('--connect_min', type=float, default=10.0,
                        help='the minimum distance to determine whether two images are connected')
    parser.add_argument('--connect_max', type=float, default=20.0,
                        help='the maximum distance to determine whether two images are connected')
    parser.add_argument('--img_dataset_num', type=int, default=10000,
                        help='the total number of the images in the datasets extracted from raw date (default: 20000)')
    parser.add_argument('--path_images', type=str, default='../dataset/carla_rawdata3/',
                        help='the path to the saved images')
    parser.add_argument('--path_train_test', type=str, default='../dataset/',
                        help='the path to the files which stored the train set and test set image names')
    parser.add_argument('--load_encoder', action='store_true',
                        help='load the encoder weights')
    parser.add_argument('--test_encoder', action='store_true',
                        help='whether test the encoder')
    parser.add_argument('--finetune', action='store_true', default=True,
                        help='load the imagenet pretrained weights')
    parser.add_argument('--connect_threshold', type=float, default=0.952,
                        help='the threshold to determine whether the two observations are connected')
    parser.add_argument('--constructgraph_topK', type=int, default=80,
                        help='exam the nearest 2K nodes when contructing the new graph')
    parser.add_argument('--sim_threshold', type=float, default=11.4,
                        help='the threshold of similarity to determine whether the two observations belong to one node')
    parser.add_argument('--max_node_dis', type=float, default=10.0,
                        help='the expected maximum L2 distance of two observations from connected nodes, \
                        regularized the cnn to maximize the distance of two observations from connected nodes')
    parser.add_argument('--node_dis_reg', type=float, default=0.04,
                        help='the regularization of the distance of observations. Regularize the cnn to minimize the distance of observations from one node,\
                        and maximize the distance of observations from connected nodes.')
    
    # The configure of GAT
    parser.add_argument('--att_out', type=int, default=256,
                        help='the dimension of the attention output layer')
    parser.add_argument('--gat_dropout', type=float, default=0.6,
                        help='Dropout rate (1 - keep probability) of GAT.')
    parser.add_argument('--nb_heads', type=int, default=4,
                        help='Number of head attentions.')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Alpha for the leaky_relu.')


    parser.add_argument('--gpu', default=[0], help='the avaliable gpu ids')
    parser.add_argument('--out_dir', type=str, default='./weights', help='the dir to save the weights')

    # RL
    parser.add_argument('--experiment-name', type=str, default='debug')
    parser.add_argument('--agent', type=str, default='a2c', help='the reinforcement learning algorithm')
    parser.add_argument('--rl_config', type=str, default='./DRL/config/a2c.yaml',
                        help='path to config file')
    parser.add_argument('--starting-port', type=int, default=2000,
                        help='starting_port')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save_interval', type=int, default=200,
                        help='save interval, one save per n updates (default: 10)')
    parser.add_argument('--video-interval', type=int, default=100,
                        help='create a visualization of the agent behavior every number of episodes')
    parser.add_argument('--save-interval', type=int, default=1000,
                        help='interval to save a checkpoint, one save per n updates (default: 1000)')
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--save-dir', default='./outputs',
                        help='directory to save models, logs and videos')
    parser.add_argument('--resume-training', default=None,
                        help='checkpoint file to resume training from')
    parser.add_argument('--load_agent_gat', action='store_true',
                        help='whether load the policy network and the GAT network')
    args = parser.parse_args()
    return args
