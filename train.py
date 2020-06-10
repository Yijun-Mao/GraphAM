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
from IPython import embed
from DRL.carla.tcp import TCPConnectionError
from DRL.carla_logger import setup_carla_logger

import traceback
import argparse
from DRL.envs_manager import make_vec_envs
from DRL.storage import RolloutStorage
from DRL.utils import get_vec_normalize, save_modules, load_modules

import datetime
from tensorboardX import SummaryWriter

from config import get_args
from DRL.observation_utils import CarlaObservationConverter
from DRL.action_utils import CarlaActionsConverter
from DRL.env import CarlaEnv
from DRL.vec_env.util import dict_to_obs, obs_to_dict

from model import Navigation

def get_config_and_checkpoint(args):
    config_dict = None

    if args.rl_config:
        config_dict = load_config_file(args.rl_config)

    if config_dict is None:
        print("ERROR: --config or --resume-training flag is required.")
        exit(1)

    config = namedtuple('Config', config_dict.keys())(*config_dict.values())
    return config


def load_config_file(filename):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)

        # To be careful with values like 7e-5
        config['lr'] = float(config['lr'])
        config['eps'] = float(config['eps'])
        config['alpha'] = float(config['alpha'])
        return config


def set_random_seeds(args, config):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if args.gpu:
        torch.cuda.manual_seed(config.seed)
    # TODO: Set CARLA seed (or env seed)


def main():
    config = None
    args = get_args()

    gpus = ''
    for ids in args.gpu:
        gpus += str(ids)
        gpus += ','
    print(gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    config = get_config_and_checkpoint(args)

    set_random_seeds(args, config)
    eval_log_dir = args.save_dir + "_eval"
    try:
        os.makedirs(args.save_dir)
        os.makedirs(eval_log_dir)
    except OSError:
        pass

    now = datetime.datetime.now()
    experiment_name = args.experiment_name + \
        '_' + now.strftime("%Y-%m-%d_%H-%M-%S")

    # Create checkpoint file
    save_dir_model = os.path.join(args.save_dir, 'model', experiment_name)
    save_dir_config = os.path.join(args.save_dir, 'config', experiment_name)
    logger = setup_carla_logger(args.save_dir, experiment_name)
    try:
        os.makedirs(save_dir_model)
        os.makedirs(save_dir_config)
    except OSError as e:
        logger.error(e)
        exit()

    if args.rl_config:
        shutil.copy2(args.rl_config, save_dir_config)

    # Tensorboard Logging
    writer = SummaryWriter(os.path.join(
        args.save_dir, 'tensorboard', experiment_name))

    # Logger that writes to STDOUT and a file in the save_dir

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    norm_reward = not config.no_reward_norm
    norm_obs = not config.no_obs_norm

    assert not (config.num_virtual_goals > 0) or (
        config.reward_class == 'SparseReward'), 'Cant use HER with dense reward'
    obs_converter = CarlaObservationConverter(
        h=args.img_height, w=args.img_width, rel_coord_system=config.rel_coord_system)
    action_converter = CarlaActionsConverter(config.action_type)
    envs = make_vec_envs(obs_converter, action_converter, args.starting_port, config.seed, config.num_processes,
                         config.gamma, device, config.reward_class, num_frame_stack=1, subset=config.experiments_subset,
                         norm_reward=norm_reward, norm_obs=norm_obs, apply_her=config.num_virtual_goals > 0,
                         video_every=args.video_interval, video_dir=os.path.join(args.save_dir, 'video', experiment_name))

    agent = Navigation(args, envs.action_space, config)


    if args.load_agent_gat:
        agent.loadmodel(args.out_dir, 5000)
        print("Loading models")

    rollouts = RolloutStorage(config.num_steps, config.num_processes,
                              envs.observation_space, envs.action_space, 
                              config.num_virtual_goals, config.rel_coord_system, obs_converter)

    obs = envs.reset()
    # Save the first observation
    obs = obs_to_dict(obs)
    rollouts.obs = obs_to_dict(rollouts.obs)
    for k in rollouts.obs:
        rollouts.obs[k][rollouts.step].copy_(obs[k])
    rollouts.obs = dict_to_obs(rollouts.obs)
    rollouts.to(device)

    start = time.time()

    total_steps = 0
    total_episodes = 0
    total_reward = 0
    # write_path = 0

    episode_reward = torch.zeros(config.num_processes)

    for j in range(config.num_updates):

        for step in range(config.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = agent.act(
                    rollouts.get_obs(step))

            # Observe reward and next obs
            obs, reward, done, info = envs.step(action)
            # if write_path% 20 == 0:
            #     _position = obs['v'][-1].cpu().detach().numpy()
            #     current_pos = _position[0:2]
            #     target_pos = _position[-3:-1]
            #     with open('./weights/drl_path/step_{}.txt'.format(write_path), 'a') as f:
            #         f.write("{} {} {} {}\n".format(current_pos[0], current_pos[1], target_pos[0], target_pos[1]))

            # For logging purposes
            carla_rewards = torch.tensor(
                [i['carla-reward'] for i in info], dtype=torch.float)
            episode_reward += carla_rewards
            total_reward += carla_rewards.sum().item()
            total_steps += config.num_processes

            if done.any():
                # write_path += 1
                total_episodes += done.sum()
                torch_done = torch.tensor(done.astype(int)).byte()
                mean_episode_reward = episode_reward[torch_done].mean().item()
                logger.info('{} episode(s) finished with reward {}'.format(
                    done.sum(), mean_episode_reward))
                writer.add_scalar('train/mean_ep_reward_vs_steps',
                                  mean_episode_reward, total_steps)
                writer.add_scalar('train/mean_ep_reward_vs_episodes',
                                  mean_episode_reward, total_episodes)
                episode_reward[torch_done] = 0

            # If done then clean the history of observations.
            masks = torch.FloatTensor(1-done)

            rollouts.insert(obs, action,action_log_prob, value, reward, masks.unsqueeze(-1))

        if config.num_virtual_goals > 0:
            rollouts.apply_her(config.num_virtual_goals,
                               device, beta=config.beta)

        with torch.no_grad():
            next_value = agent.get_value(rollouts.get_obs(-1)).detach()

        rollouts.compute_returns(
            next_value, config.use_gae, config.gamma, config.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts, j)

        rollouts.after_update()

        if j % args.save_interval == 0 and args.out_dir != "" and config.agent != 'forward':
            # save_path = os.path.join(save_dir_model, str(j) + '.pth.tar')
            # save_modules(agent.optimizer, agent.model, args, config, save_path)
            agent.savemodel(args.out_dir, j)

        total_num_steps = (j + 1) * config.num_processes * config.num_steps

        if j % args.log_interval == 0:

            # Logging to the stdout/our logs
            end = time.time()
            logger.info('------------------------------------')
            logger.info('Episodes {}, Updates {}, num timesteps {}, FPS {}, value_loss {}, action_loss {}, dist_entropy {}'
                        .format(total_episodes, j + 1, total_num_steps, total_num_steps / (end - start), value_loss, action_loss, dist_entropy))
            logger.info('------------------------------------')

            # Logging to tensorboard
            writer.add_scalar('train/cum_reward_vs_steps',
                              total_reward, total_steps)
            writer.add_scalar('train/cum_reward_vs_updates', total_reward, j+1)

            if config.agent in ['a2c', 'acktr', 'ppo']:
                writer.add_scalar('debug/value_loss_vs_steps',
                                  value_loss, total_steps)
                writer.add_scalar(
                    'debug/value_loss_vs_updates', value_loss, j+1)
                writer.add_scalar('debug/action_loss_vs_steps',
                                  action_loss, total_steps)
                writer.add_scalar(
                    'debug/action_loss_vs_updates', action_loss, j+1)
                writer.add_scalar('debug/dist_entropy_vs_steps',
                                  dist_entropy, total_steps)
                writer.add_scalar(
                    'debug/dist_entropy_vs_updates', dist_entropy, j+1)

            # Sample the last reward
            writer.add_scalar(
                'debug/sampled_normalized_reward_vs_steps', reward.mean(), total_steps)
            writer.add_scalar(
                'debug/sampled_normalized_reward_vs_updates', reward.mean(), j+1)
            writer.add_scalar('debug/sampled_carla_reward_vs_steps',
                              carla_rewards.mean(), total_steps)
            writer.add_scalar(
                'debug/sampled_carla_reward_vs_updates', carla_rewards.mean(), j+1)

        if (args.eval_interval is not None and j % args.eval_interval == 0):
            eval_envs = make_vec_envs(
                args.env_name, args.starting_port, obs_converter, args.x +
                config.num_processes, config.num_processes,
                config.gamma, eval_log_dir, config.add_timestep, device, True)

            vec_norm = get_vec_normalize(eval_envs)
            if vec_norm is not None:
                vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_masks = torch.zeros(config.num_processes, 1, device=device)

            while len(eval_episode_rewards) < 10:
                with torch.no_grad():
                    _, action, _ = agent.act(obs, deterministic=True)

                # Obser reward and next obs
                carla_obs, reward, done, infos = eval_envs.step(action)

                eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                                for done_ in done])
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])

            eval_envs.close()

            logger.info(" Evaluation using {} episodes: mean reward {:.5f}\n".
                        format(len(eval_episode_rewards),
                               np.mean(eval_episode_rewards)))


if __name__ == "__main__":
    main()
