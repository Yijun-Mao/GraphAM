import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np

from cnn import Encoder
import sys
import scipy.sparse as sp
from tqdm import tqdm

sys.path.append('../')
from config import get_args
from utils.utils import progress_bar
from utils.makedataset import make_cnn_dataset
from utils.dataset import EncoderData, SimulateConstructGraphData
from utils.visualizegraphinmap import draw_node_edge


class ConstructGraph(object):
    '''
    The ConstructGraph module is uesd to construct the graph based on the input images and Encoder
    '''
    def __init__(self, args):
        super(ConstructGraph).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.encoder = Encoder(args)
        self.feature_size = self.encoder.cnn.feature_size
        self.args = args

        self.all_node = []
        self.all_edge = []

        self.adj = None

        self.encoder.load_model(os.path.join(args.out_dir, 'min_{}_max_{}'.format(args.connect_min, args.connect_max), 'encoder_new', 'best'))
    
    def findnewnode(self, img, searchall=True):
        '''
        Determine whether the input image can represents a new node
        :param img: the input image which should have the size: 3 x args.width x args.height
        :param searchall: if True, search all nodes to find all the connection. If False, get the first connecting node.(search start from the last one)
        '''
        feature_node = self.encoder.getobservefromimg(img)
        if len(self.all_node) == 0:
            # There is no node before, construct the first node
            self.all_node.append(feature_node.cpu().data.numpy())
            return True
        else:
            tmp_edges = {}
            idx = list(range(len(self.all_node)))[::-1]
            for i in idx:
                p_connect = self.encoder.predictconnect(self.all_node[i], feature_node.cpu().data.numpy())
                
                if p_connect[1] >= self.args.connect_threshold:
                    if searchall:
                        tmp_edges[p_connect[1]] = [i, len(self.all_node)]
                    else:
                        self.all_node.append(feature_node.cpu().data.numpy())
                        self.all_edge.append([i, len(self.all_node)])
                        return True
            
            if len(tmp_edges) > 0:
                tmp_edge_sort = sorted(tmp_edges.items(), key=lambda x:x[0], reverse=True)
                if len(tmp_edge_sort) > self.args.constructgraph_topK:
                    tmp_edge_sort = tmp_edge_sort[0:self.args.constructgraph_topK]
                for p_edges in tmp_edge_sort:
                    self.all_edge.append(p_edges[1])
                self.all_node.append(feature_node.cpu().data.numpy())

                return True

        return False

    def consgraph(self):
        '''
        construct the contructed graph after exploring from all_node and all_edge. The adjacent matrix will be saved as sparse matrix
        '''
        length_node = len(self.all_node)
        assert length_node>0, 'There is no node now'

        data = np.ones(int(len(self.all_edge)*2))
        row=[]
        col=[]
        for edge in self.all_edge:
            row.append(edge[0])
            col.append(edge[1])
            
            row.append(edge[1])
            col.append(edge[0])
        self.adj = sp.coo_matrix((data, (row, col)), shape=(length_node, length_node))

    def getgraph(self, sparse=False):
        '''
        Return the nodes and the adjacent matrix
        :param sparse: return the sparse matrix or not
        return: all the nodes and the adjacent matrix
        '''
        if self.adj is None:
            self.consgraph()
        if not sparse:
            return self.all_node, self.adj.toarray()
        else:
            return self.all_node, self.adj

    def savegraph(self, path):
        '''
        Save the nodes and adjacent matrix
        :param path: the path to save the graph
        '''
        if not os.path.exists(path):
            os.makedirs(path)
        print("Saving graph in "+path)
        sp.save_npz(os.path.join(path, 'adjacent_sparse.npz'), self.adj)
        np.save(os.path.join(path, 'nodes.npy'), self.all_node)
    
    def loadgraph(self, path):
        '''
        Load the graph from the folder
        param path: the path of the folder which the graph are saved
        '''
        assert os.path.isfile(os.path.join(
            path, 'adjacent_sparse.npz')), '{} is not existed, please check the path'.format(os.path.join(path, 'adjacent_sparse.npz'))
        assert os.path.isfile(os.path.join(
            path, 'nodes.npy')), '{} is not existed, please check the path'.format(os.path.join(path, 'nodes.npy'))
        
        print("Loading graph from "+ path)

        self.adj = sp.load_npz(os.path.join(path, 'adjacent_sparse.npz'))
        self.all_node = np.load(os.path.join(path, 'nodes.npy'))


def SimulateGraph(args):
    makegraph = ConstructGraph(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    simulateloader = DataLoader(SimulateConstructGraphData(args.path_images, 
                            target_transform=transforms.Compose([transforms.Resize(args.img_width),
                                                                transforms.ToTensor(),
                                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])),
                            batch_size=1, shuffle=False, num_workers=4)

    coords = []
    simulateloader = tqdm(simulateloader)
    n=0
    for img, coord_heading in simulateloader:
        if makegraph.findnewnode(img, searchall=True):
            # Succesful add node
            coords.append(coord_heading)
        n+=1
        simulateloader.set_description("Now get {} nodes, discard {} images".format(len(coords), n-len(coords)))

    makegraph.consgraph()
    all_nodes, adjacent = makegraph.getgraph(sparse=False)
    assert len(all_nodes) == len(coords), "The nodes and the coords are not equal"
    print("There are {} nodes.".format(len(all_nodes)))
    makegraph.savegraph(os.path.join(args.out_dir, 'min_{}_max_{}'.format(args.connect_min, args.connect_max), 'graph'))
    np.save(os.path.join(args.out_dir, 'min_{}_max_{}'.format(args.connect_min, args.connect_max), 'graph', 'coordinate.npy'), coords)
    # makegraph.loadgraph(os.path.join(args.out_dir, 'min_{}_max_{}'.format(args.connect_min, args.connect_max), 'graph'))
    # all_nodes, adjacent = makegraph.getgraph(sparse=False)
    # coords = np.load(os.path.join(args.out_dir, 'min_{}_max_{}'.format(args.connect_min, args.connect_max), 'graph', 'coordinate.npy'),allow_pickle=True)
    print("Visualing the graph")
    draw_node_edge(adjacent, coords, os.path.join(args.out_dir, 'min_{}_max_{}'.format(args.connect_min, args.connect_max), 'graph'))
    # 1766 nodes

if __name__ == '__main__':
    args = get_args()
    gpus = ''
    for ids in args.gpu:
        gpus+=str(ids)
        gpus+=','
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    SimulateGraph(args)
        