from env.chooseenv import make
from rlcard.agents.dqn_agent_two import DQNAgent
from agent.random.random_agent import RandomAgent
import os
from utils.utils import run, reorganize, reorganize2
import torch

env = make("chessandcard-mahjong_v3")
env.set_seed(42)
agents = []

observation_shape = env.input_dimension['player_0']['observation'].shape
action_type = env.joint_action_space[0][0]
save_dir = os.path.abspath(__file__)
device = 'cpu'

Random_agent = RandomAgent(action_type, False)
DQN_agent = DQNAgent(num_actions=action_type.n, state_shape=observation_shape,
                     mlp_layers=[64, 64], device=device)
agents.append(DQN_agent)
for _ in range(env.n_player - 1):
    agents.append(Random_agent)

trajectories, payoffs = run(env, agents)
new_trajectories = reorganize2(trajectories, payoffs)
print(len(trajectories[0]), len(trajectories[1]), len(trajectories[2]), len(trajectories[3]))
print(len(new_trajectories[0]), len(new_trajectories[1]), len(new_trajectories[2]), len(new_trajectories[3]))
