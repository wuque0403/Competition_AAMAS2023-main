from env.chooseenv import make
from rlcard.agents.PPO_agent_two import PPOAgent
from tensorboardX import SummaryWriter
from agent.random.random_agent import RandomAgent
import os
from utils.utils import run, reorganize, reorganize2
import argparse
import torch
import numpy as np
'''
env = make("chessandcard-mahjong_v3")
obs = env.reset()[0]['obs']
# obs = env.all_observes[0]['obs']
print(obs)
print(env.env_core.agent_selection)
action_index = list(np.where(obs['action_mask'] == 1))[0]
each = [0] * env.joint_action_space[0][0].n
each[action_index[0]] = 1
print(each)
all_observe, reward, done, _, _ = env.step([[each], [], [], []])
print(obs['action_mask'].shape[0], action_index)
print(reward, done)
print(env.env_core.agent_selection)
print(all_observe)
'''

env = make("chessandcard-mahjong_v3")
env.set_seed(42)
agents = []

observation_shape = env.input_dimension['player_0']['observation'].shape
action_type = env.joint_action_space[0][0]
save_dir = os.path.abspath(__file__)
PPO_agent_learner = PPOAgent(observation_shape, action_type.n, save_dir)  # 用于学习
PPO_agent_actor = PPOAgent(observation_shape, action_type.n, save_dir)  # 用于生成数据
Random_agent = RandomAgent(action_type, False)
agents.append(PPO_agent_actor)
for _ in range(env.n_player - 1):
   agents.append(Random_agent)

trajectories, payoff = run(env, agents)
new_trajectories = reorganize(trajectories, payoff)
new_trajectories2 = reorganize2(trajectories, payoff)

'''print(new_trajectories[0][0][0]['observation'].shape)
obs = new_trajectories[0][0][0]['observation']
obs_v = torch.FloatTensor(obs).unsqueeze(0)
print(PPO_agent_learner.critic_net(obs_v))
states, _, _, _ = PPO_agent_learner.unpack(new_trajectories[0])
print(len(new_trajectories[0]), len(states))'''
print(len(trajectories[0]), len(trajectories[1]), len(trajectories[2]), len(trajectories[3]))
print(len(new_trajectories[0]), len(new_trajectories[1]), len(new_trajectories[2]), len(new_trajectories[3]))
print(len(new_trajectories2[0]), len(new_trajectories2[1]), len(new_trajectories2[2]), len(new_trajectories2[3]))
print(new_trajectories2[0][0])
action_index = list(np.where(new_trajectories2[0][0][0]['action_mask'] == 1))[0].tolist()
print(action_index, type(action_index))
#for _ in range(100):
    #trajectories, payoff = run(env, agents)
