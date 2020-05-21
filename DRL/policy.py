import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from distributions import Categorical, DiagGaussian

class Policy(nn.Module):
    def __init__(self, obs_space, guide_space, action_space, num_layers=3, n_latent_var=128, mode="Box"):
        super(Policy, self).__init__()
        self.obs_space = obs_space
        self.guide_space = guide_space
        self.action_space = action_space
        self.mode = mode

        self.actor = [nn.Linear(obs_space+guide_space, 512), nn.Tanh(),
                      nn.Linear(512, n_latent_var), nn.Tanh()]
        self.critic = [nn.Linear(obs_space+guide_space, 512), nn.Tanh(),
                       nn.Linear(512, n_latent_var), nn.Tanh()]
        for i in range(num_layers-2):
            self.actor.append(nn.Linear(n_latent_var, n_latent_var))
            self.actor.append(nn.Tanh())
            self.critic.append(nn.Linear(n_latent_var, n_latent_var))
            self.critic.append(nn.Tanh())
        # self.actor.append(nn.Linear(n_latent_var, action_space))
        self.critic.append(nn.Linear(n_latent_var, 1))

        if mode == "Discrete":
            self.dist = Categorical(n_latent_var, action_space)
        elif mode == "Box":
            self.dist = DiagGaussian(n_latent_var, action_space)
        else:
            raise NotImplementedError

    def act(self, observation, guide, deterministic):
        concat = torch.cat([observation, guide], dim=-1)
        actor_features = self.actot(concat)
        value = self.critic(concat)

        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        
        action_log_probs = dist.log_probs(action)
        # dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs
    
    def get_value(self, observation, guide):
        concat = torch.cat([observation, guide], dim=-1)
        value = self.critic(concat)
        return value

    def evaluate_actions(self, observation, guide, action):
        concat = torch.cat([observation, guide], dim=-1)
        actor_features = self.actot(concat)
        value = self.critic(concat)

        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy
