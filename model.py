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
import DRL.agents as agents
from DRL.policy import Policy

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

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
        self.node_features = None
        self.coords = None

        self.config = config

        self.agent = Policy(self.feature_size, self.guide_space, action_space).to(self.device)

        self.value_loss_coef = self.config.value_loss_coef
        self.entropy_coef = self.config.entropy_coef
        

        # Epsilon Greedy Values
        self.eps_curr = eps_greedy_start
        self.eps_greedy_decay = eps_greedy_decay

        self.max_grad_norm = self.config.max_grad_norm
        self.lr = self.config.lr

        if config.agent == 'ppo':
            self.optimizer = optim.Adam([{'params': self.agent.parameters(), 'lr': self.lr, 'eps': self.config.eps}, {
                'params': self.gat.parameters(), 'lr': self.lr, 'eps': self.config.eps}])
            # self.agent_optim = optim.Adam(self.agent.model.parameters(), lr, )
            # self.gat_optim = optim.Adam(self.gat.parameters(), lr, weight_decay=weight_decay)
        elif config.agent == 'a2c':
            self.optimizer = optim.RMSprop([{'params': self.agent.parameters(), 'lr': self.lr, 'eps': self.config.eps, 'alpha': self.config.alpha}, {
                'params': self.gat.parameters(), 'lr': self.lr, 'eps': self.config.eps, 'alpha': self.config.alpha}])
            # self.agent_optim = optim.RMSprop(self.agent.model.parameters(), lr, eps=eps, alpha=alpha)
        else:
            raise NotImplementedError

    def locateingraph(self, observation, L=1, groundtruth=False):
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
                return similarities[L//2][0]
            else:
                if len(observation.shape)==1:
                    observation = np.expand_dims(observation,axis=0)
                all_nodes = np.expand_dims(self.all_nodes,axis=1).repeat(observation.shape[0],axis=1) 
                similarity = np.linalg.norm(all_nodes - observation, ord=2, axis=-1)
                return np.argmin(similarity, axis=0)
        else:
            if self.coords is None:
                coords = np.load(os.path.join(self.args.out_dir, 'min_{}_max_{}'.format(
                    self.args.connect_min, self.args.connect_max), 'graph', 'coordinate.npy'))
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
        :param nodes: the nodes in the graph (torch.tensor)
        :param adj: the adjacent matrix of the graph (torch.tensor)
        return: the new features after aggregating
        '''
        nodes = nodes.to(self.device)
        adj = adj.to(self.device)
        output = self.gat(nodes, adj)
        return output

    def act(self, inputs, deterministic=False, groundtruth=True):
        img = inputs['img'].astype(np.float)
        img = torch.from_numpy(img)
        observation = self.encoder.getobservefromimg(img)
        current_i = self.locateingraph(observation.cpu().detach().numpy(), L=1)

        # if updatetarget:
            # target = inputs['v'].astype(np.float)
            # target = torch.from_numpy(target)
            # target_obs = self.encoder.getobservefromimg(target)
            # self.target_i_act = self.locateingraph(target_obs, L=3)
            # self.target_i_act = self.locateingraph(inputs['v'], groundtruth=groundtruth)
        self.target_i_act = self.locateingraph(inputs['v'], groundtruth=groundtruth)

        guide = self.node_features[self.target_i_act] - self.node_features[current_i]

        value, action, action_log_prob = self.agent.act(
            observation, guide, deterministic)
        
        return value, action, action_log_prob

    def get_value(self, inputs, groundtruth=True):
        img = inputs['img'].astype(np.float)
        img = torch.from_numpy(img)
        observation = self.encoder.getobservefromimg(img)
        current_i = self.locateingraph(observation.cpu().detach().numpy(), L=1)

        # if updatetarget:
            # target = inputs['v'].astype(np.float)
            # target = torch.from_numpy(target)
            # target_obs = self.encoder.getobservefromimg(target)
            # self.target_i_value = self.locateingraph(target_obs, L=3)
        target_i_value = self.locateingraph(inputs['v'], groundtruth=groundtruth)

        guide = self.node_features[target_i_value] - self.node_features[current_i]
        return self.agent.get_value(observation, guide)

    def update(self, rollouts, groundtruth=True):
        if self.config.agent == 'a2c':
            self.eps_curr = max(0.0, self.eps_curr - self.eps_greedy_decay)
            # obs_shape = {k: r.size()[2:] for k, r in rollouts.obs.items()}
            # rollouts_flatten = {k: r[:-1].view(-1, *obs_shape[k]) for k, r in rollouts.obs.items()}
            
            num_steps, num_processes, _ = rollouts.rewards.size()
            observation = self.encoder.getobservefromimg(rollouts['img'])
            current_i = self.locateingraph(observation.cpu().detach().numpy(), L=1)

            target_i_act = self.locateingraph(rollouts['v'], groundtruth=groundtruth)

            guide = self.node_features[target_i_act] - self.node_features[current_i]

            values, action_log_probs, dist_entropy = self.agent.evaluate_actions(
                observation,
                guide,
                rollouts.actions.view(-1, self.action_space))

            values = values.view(num_steps, num_processes, 1)
            action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

            advantages = rollouts.returns[:-1] - values
            value_loss = advantages.pow(2).mean()

            action_loss = -(advantages.detach() * action_log_probs).mean()

            # if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            #     # Sampled fisher, see Martens 2014
            #     self.model.zero_grad()
            #     pg_fisher_loss = -action_log_probs.mean()

            #     value_noise = torch.randn(values.size())
            #     if values.is_cuda:
            #         value_noise = value_noise.cuda()

            #     sample_values = values + value_noise
            #     vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            #     fisher_loss = pg_fisher_loss + vf_fisher_loss
            #     self.optimizer.acc_stats = True
            #     fisher_loss.backward(retain_graph=True)
            #     self.optimizer.acc_stats = False

            self.optimizer.zero_grad()
            (value_loss * self.value_loss_coef + action_loss -
            dist_entropy * self.entropy_coef).backward()

            # if self.acktr == False:
            nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)

            self.optimizer.step()

            return value_loss.item(), action_loss.item(), dist_entropy.item()


    def savemodel(self, path):
        print('Storing agent model to: {}'.format(path))
        torch.save({'state_dict': self.agent.state_dict(),
                    'config': dict(self.config._asdict())  # Save as a Python dictionary
                    }, os.path.join(path, 'agent.pt'))
        
        print('Storing GAT model to: {}'.format(path))
        torch.save(self.gat.state_dict(), os.path.join(path, 'GAT.pt'))
        
    def loadmodel(self, path):
        print("Loading weights from " + path)
        checkpoints = torch.load(os.path.join(path, 'agent.pt'))
        self.agent.load_state_dict(checkpoints['state_dict'])
        self.gat.load_state_dict(torch.load(os.path.join(path, 'GAT.pt')))
        



