import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import get_device, set_seed, reorganize
from rlcard.agents.PPO_agent import PPOAgent
import argparse
import os


def train(args):
    device = get_device()
    set_seed(args.seed)
    env = rlcard.make(args.env, config={'seed': args.seed})

    agents = []
    PPO_agent = PPOAgent(env.state_shape[0], env.num_actions, args.save_dir, device)
    PPO_agent.load(args.save_dir)
    Random_agent = RandomAgent(env.num_actions)
    agents.append(PPO_agent)
    for _ in range(1, env.num_players):
        agents.append(Random_agent)

    env.set_agents(agents)
    trajectories = []
    while True:
        trajectory, payoff = env.run(is_training=True)
        trajectory = reorganize(trajectory, payoff)[0]
        for item in trajectory:
            if len(trajectories) < args.trajectory_size:
                trajectories.append(item)
        print("trajectories length is %d" % len(trajectories))
        if len(trajectories) < args.trajectory_size:
            continue
        best_reward = PPO_agent.train(trajectories)
        env.reset()
        if best_reward >= args.goal_reward:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="mahjong")
    parser.add_argument("--seed", type=int, default=42)

    save_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--save_dir", type=str, default=save_dir)
    parser.add_argument("--trajectory_size", type=int, default=2049)
    parser.add_argument('--goal_reward', type=float, default=0.5)
    args = parser.parse_args()
    train(args)
