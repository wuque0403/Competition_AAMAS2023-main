import numpy as np
from env.chooseenv import make
from rlcard.agents.dqn_agent_two import DQNAgent
from tensorboardX import SummaryWriter
from agent.random.random_agent import RandomAgent
import os
from utils.utils import run, reorganize2
import argparse
from collections import deque


def train(args):
    env = make(args.env)
    env.set_seed(args.seed)
    agents = []

    observation_shape = env.input_dimension['player_0']['observation'].shape
    action_type = env.joint_action_space[0][0]

    Random_agent = RandomAgent(action_type, is_act_continuous=False)
    DQN_agent = DQNAgent(replay_memory_size=args.memory_size, num_actions=action_type.n, mlp_layers=[128, 128],
                         state_shape=observation_shape, device=args.device, save_path=args.save_dir)
    agents.append(DQN_agent)
    for _ in range(env.n_player - 1):
        agents.append(Random_agent)

    total_rewards = deque(maxlen=100)
    writer = SummaryWriter(comment='mean_episodes_reward')
    best_mean_episodes = None
    mean_episodes_reward = None

    for episode in range(args.train_episodes):
        trajectories, payoffs = run(env, agents)
        new_trajectories = reorganize2(trajectories, payoffs)[0]
        total_rewards.append(payoffs['player_0'])
        mean_episodes_reward = float(np.mean(total_rewards))
        for ts in new_trajectories:
            DQN_agent.feed(ts)

        if best_mean_episodes is None or best_mean_episodes < mean_episodes_reward:
            DQN_agent.save_checkpoint(args.save_dir)
            if best_mean_episodes is not None:
                print("best_mean_rewards updates %.2f -> %.2f, Model saved" % (best_mean_episodes, mean_episodes_reward))
            best_mean_episodes = mean_episodes_reward
        print("best_mean_reward is %.2f, mean_100episodes_reward is %.2f" % (best_mean_episodes, mean_episodes_reward))

        if best_mean_episodes >= args.goal_reward:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='chessandcard-mahjong_v3')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    save_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--save_dir', type=str, default=save_dir)
    parser.add_argument('--memory_size', type=int, default=10000)
    parser.add_argument('--goal_reward', type=float, default=0.15)
    parser.add_argument('--train_episodes', type=int, default=20000)
    args = parser.parse_args()
    train(args)
