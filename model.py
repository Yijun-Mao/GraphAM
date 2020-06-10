import copy
import os
import time
import yaml
import shutil
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Encoder.ConstrucGraph import ConstructGraph
from Encoder.cnn import Encoder
from GAT.models import GAT
from utils.utils import *
from DRL.policy import Policy


def load_config_file(filename):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)

        # To be careful with values like 7e-5
        config['lr'] = float(config['lr'])
        config['eps'] = float(config['eps'])
        config['alpha'] = float(config['alpha'])
        return config

def get_config(args):
    config_dict = None

    if args.rl_config:
        config_dict = load_config_file(args.rl_config)

    if config_dict is None:
        print("ERROR: --config or --resume-training flag is required.")
        exit(1)

    config = namedtuple('Config', config_dict.keys())(*config_dict.values())
    return config

class Navigation(object):
    def __init__(self, 
                 args, 
                 action_space,
                 config,
                 eps_greedy_start=0.0, 
                 eps_greedy_decay=0.0001):
        super().__init__()
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.graph = ConstructGraph(args, loadencoder=False)

        self.encoder = Encoder(args)
        self.encoder.connect.eval()
        self.encoder.cnn.eval()
        self.feature_size = self.encoder.cnn.feature_size
        self.action_space = action_space

        self.gat = GAT(nfeat=self.feature_size,
                       nhid=args.att_out,
                       dropout=args.gat_dropout,
                       nheads=args.nb_heads,
                       alpha=args.alpha)
        self.gat.to(self.device)

        self.guide_space = self.gat.guide_space
        try:
            self.encoder.load_model(os.path.join(args.out_dir, 'min_{}_max_{}'.format(
                args.connect_min, args.connect_max), 'encoder_new', 'best'))
            self.graph.loadgraph(os.path.join(args.out_dir, 'min_{}_max_{}'.format(args.connect_min, args.connect_max), 'graph'))
        except OSError:
            print('The checkpoints or the constructed graph are not existed!')

        self.all_nodes, self.adjacent = self.graph.getgraph(sparse=False)
        for i in range(len(self.all_nodes)):
            self.adjacent[i,i] = 1
        self.node_features = None
        self.coords = None

        self.config = config

        if action_space.__class__.__name__ == "Discrete":
            self.num_outputs = action_space.n
            mode = "Discrete"
        elif action_space.__class__.__name__ == "Box":
            self.num_outputs = action_space.shape[0]
            mode = "Box"
        else:
            raise NotImplementedError
        # if config.action_type == "carla-original":
        #     action_num_outputs = envs.action_space.n
        # elif config.action_type == "continuous":
        #     action_num_outputs = envs.action_space.shape[0]

        self.agent = Policy(self.feature_size, self.guide_space, action_space, mode=mode).to(self.device)

        self.value_loss_coef = self.config.value_loss_coef
        self.entropy_coef = self.config.entropy_coef
        

        # Epsilon Greedy Values
        self.eps_curr = eps_greedy_start
        self.eps_greedy_decay = eps_greedy_decay

        self.max_grad_norm = self.config.max_grad_norm
        self.lr = self.config.lr

        if config.agent == 'ppo':
            self.optimizer = optim.Adam([{'params': self.agent.parameters(), 'lr': self.lr, 'eps': self.config.eps}, {
                'params': self.gat.parameters(), 'lr': self.lr*0.05, 'eps': self.config.eps}])
            # self.agent_optim = optim.Adam(self.agent.model.parameters(), lr, )
            # self.gat_optim = optim.Adam(self.gat.parameters(), lr, weight_decay=weight_decay)
        elif config.agent == 'a2c':
            self.optimizer = optim.RMSprop([{'params': self.agent.parameters(), 'lr': self.lr, 'eps': self.config.eps, 'alpha': self.config.alpha}, {
                'params': self.gat.parameters(), 'lr': self.lr*0.05, 'eps': self.config.eps, 'alpha': self.config.alpha}])
            # self.agent_optim = optim.RMSprop(self.agent.model.parameters(), lr, eps=eps, alpha=alpha)
        else:
            raise NotImplementedError

    def locateingraph(self, observation, L=10, groundtruth=False):
        '''
        Locate the observation in constructed graph. 
        Pick up top L similar nodes and select the node of median similarity score as the current node.
        :param observation: the current observation (numpy.array)
        :param groundtruth: whether to load the groudtruth position. When groundtruth is True, the observtion should be the coordination of target.
            This is used when training
        return: the [idx] of node in self.all_nodes 
        '''
        observation = np.array(observation)
        observation = np.squeeze(observation)
        if not groundtruth:
            similarities = []
            if L >1:
                for i in range(self.all_nodes.shape[0]):
                    node = self.all_nodes[i]
                    similarity = np.linalg.norm(node - observation, ord=2, axis=-1)
                    similarities.append([i, similarity])
                
                similarities = sorted(similarities, key=lambda x: x[1], reverse=False)
                node_idx = similarities[L//2][0]
                if self.coords is not None:
                    print(self.coords[node_idx])
                return node_idx
            else:
                if len(observation.shape)==1:
                    observation = np.expand_dims(observation,axis=0)
                all_nodes = np.expand_dims(self.all_nodes,axis=1).repeat(observation.shape[0],axis=1) 
                similarity = np.linalg.norm(all_nodes - observation, ord=2, axis=-1)
                node_idx = np.argmin(similarity, axis=0)
                # if self.coords is not None:
                #     print(self.coords[node_idx])
                return node_idx
        else:
            if self.coords is None:
                coords = np.load(os.path.join(self.args.out_dir, 'min_{}_max_{}'.format(
                    self.args.connect_min, self.args.connect_max), 'graph', 'coordinate.npy'), allow_pickle=True)
                self.coords = np.array([[x,y] for x,y,head in coords])
            if len(observation.shape) == 1:
                observation = np.expand_dims(observation, axis=0)
            coords = np.expand_dims(self.coords,axis=1).repeat(observation.shape[0],axis=1) 
            distances = np.linalg.norm(coords - observation, ord=2, axis=-1)
            return np.argmin(distances, axis=0)
            # min_dis = -1
            # min_i = -1
            # for i, coord in enumerate(self.coords):
            #     distance = computedistance2points([observation[0], observation[1]], [coord[0],coord[1]])
            #     if min_dis == -1 or min_dis > distance:
            #         min_dis=distance
            #         min_i = i
            # return min_i

    def aggregatefeature(self, nodes, adj):
        '''
        Aggregate the features in graph with GAT
        :param nodes: the nodes in the graph (numpy.array)
        :param adj: the adjacent matrix of the graph (numpy.array)
        return: the new features after aggregating
        '''
        nodes = torch.FloatTensor(nodes).to(self.device)
        adj = torch.FloatTensor(adj).to(self.device).squeeze()
        output = self.gat(nodes, adj)
        # self.node_features = output
        return output

    def act(self, inputs, deterministic=False, groundtruth=True):
        img = inputs['img']
        # img = torch.from_numpy(img)
        target = inputs['v']
        current = target[:,0:2].cpu().numpy()
        target = target[:,-3:-1].cpu().numpy()
        # print("target ",target)
        # print("current ",current)
        observation = self.encoder.getobservefromimg(img)
        # current_i = self.locateingraph(observation.cpu().detach().numpy(), L=1)
        current_i = self.locateingraph(current, groundtruth=groundtruth)
        self.node_features = self.aggregatefeature(self.all_nodes, self.adjacent)
        # if updatetarget:
            # target = inputs['v'].astype(np.float)
            # target = torch.from_numpy(target)
            # target_obs = self.encoder.getobservefromimg(target)
            # self.target_i_act = self.locateingraph(target_obs, L=3)
            # self.target_i_act = self.locateingraph(inputs['v'], groundtruth=groundtruth)
        self.target_i_act = self.locateingraph(target, groundtruth=groundtruth)

        guide = self.node_features[self.target_i_act] - self.node_features[current_i]

        value, action, action_log_prob = self.agent.act(
            observation, guide, self.eps_curr, deterministic)
        
        return value, action, action_log_prob

    def get_value(self, inputs, groundtruth=True):
        img = inputs['img']
        # img = torch.from_numpy(img)
        target = inputs['v']
        current = target[:,0:2].cpu().numpy()
        target = target[:,-3:-1].cpu().numpy()
        observation = self.encoder.getobservefromimg(img)
        # current_i = self.locateingraph(observation.cpu().detach().numpy(), L=1)
        current_i = self.locateingraph(current, groundtruth=groundtruth)

        # if updatetarget:
            # target = inputs['v'].astype(np.float)
            # target = torch.from_numpy(target)
            # target_obs = self.encoder.getobservefromimg(target)
            # self.target_i_value = self.locateingraph(target_obs, L=3)
        target_i_value = self.locateingraph(target, groundtruth=groundtruth)
        self.node_features = self.aggregatefeature(self.all_nodes, self.adjacent)

        guide = self.node_features[target_i_value] - self.node_features[current_i]
        return self.agent.get_value(observation, guide)

    def update(self, rollouts, step, groundtruth=True):
        if self.config.agent == 'a2c':
            self.eps_curr = max(0.0, self.eps_curr - self.eps_greedy_decay)
            obs_shape = {k: r.size()[2:] for k, r in rollouts.obs.items()}
            # rollouts_flatten = {k: r[:-1].view(-1, *obs_shape[k]) for k, r in rollouts.obs.items()}
            rollouts_flatten = {}
            for k,r in rollouts.obs.items():
                r = r[:-1].view(-1, *obs_shape[k])
                # if k == 'v':
                #     r = r[:, -3:-1]
                rollouts_flatten[k] = r
                    
            action_shape = rollouts.actions.size()[-1]
            num_steps, num_processes, _ = rollouts.rewards.size()
            observation = self.encoder.getobservefromimg(rollouts_flatten['img'])
            # current_i = self.locateingraph(observation.cpu().detach().numpy(), L=1)
            _position = rollouts_flatten['v'].cpu().detach().numpy()
            current_pos = _position[:,0:2]
            target_pos = _position[:,-3:-1]
            # self.savePath(current_pos, target_pos, step)
            current_i = self.locateingraph(current_pos, groundtruth=groundtruth)

            target_i_act = self.locateingraph(target_pos, groundtruth=groundtruth)
            
            self.node_features = self.aggregatefeature(self.all_nodes, self.adjacent)

            guide = self.node_features[target_i_act] - self.node_features[current_i]

            values, action_log_probs, dist_entropy = self.agent.evaluate_actions(
                observation,
                guide,
                rollouts.actions.view(-1, action_shape))

            values = values.view(num_steps, num_processes, 1)
            action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

            advantages = rollouts.returns[:-1] - values
            value_loss = advantages.pow(2).mean()

            action_loss = -(advantages.detach() * action_log_probs).mean()


            self.optimizer.zero_grad()
            (value_loss * self.value_loss_coef + action_loss -
            dist_entropy * self.entropy_coef).backward()

            # if self.acktr == False:
            nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)

            self.optimizer.step()

            return value_loss.item(), action_loss.item(), dist_entropy.item()

    def savePath(self, current, target, step):
        if current.shape[0] > 5:
            print("writing path")
            with open('./weights/drl_path/step_{}.txt'.format(step), 'w') as f:
                for i in range(current.shape[0]):
                    f.write("{} {}\n".format(current[i,0], current[i, 1]))
                f.write("{} {}".format(target[0,0], target[0, 1]))

    def savemodel(self, path, step=0):
        if not os.path.exists(os.path.join(path,'drl')):
            os.makedirs(os.path.join(path,'drl'))
        print('Storing agent model to: {}'.format(path))
        torch.save({'state_dict': self.agent.state_dict(),
                    'config': dict(self.config._asdict())  # Save as a Python dictionary
                    }, os.path.join(path, 'drl',  self.config.agent + '_agent_{}.pt'.format(step)))
        
        print('Storing GAT model to: {}'.format(path))
        torch.save(self.gat.state_dict(), os.path.join(path, 'drl', self.config.agent + '_GAT_{}.pt'.format(step)))
        
    def loadmodel(self, path, step):
        print("Loading weights from " + path)
        checkpoints = torch.load(os.path.join(path, 'drl',  self.config.agent + '_agent_{}.pt'.format(step)), map_location=lambda storage, loc: storage)
        self.agent.load_state_dict(checkpoints['state_dict'])
        self.gat.load_state_dict(torch.load(os.path.join(path,'drl', self.config.agent + '_GAT_{}.pt'.format(step))))
        



