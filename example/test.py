from env.chooseenv import make
from rlcard.agents.PPO_agent_two import PPOAgent
from tensorboardX import SummaryWriter
from agent.random.random_agent import RandomAgent
import os
from utils.utils import run, reorganize
import argparse
import torch


def test_net(env, agent_one, agent_two, count=100):
    agents = [agent_one]
    for _ in range(env.n_player - 1):
        agents.append(agent_two)
    total_payoff = 0
    for _ in range(count):
        _, payoffs = run(env, agents)
        total_payoff += payoffs['player_0']
    return total_payoff / count


def train(args):
    env = make(args.env)
    env.set_seed(args.seed)
    agents = []

    observation_shape = env.input_dimension['player_0']['observation'].shape
    action_type = env.joint_action_space[0][0]
    PPO_agent_learner = PPOAgent(observation_shape, action_type.n, args.save_dir, args.device)  # 用于学习
    PPO_agent_actor = PPOAgent(observation_shape, action_type.n, args.save_dir, args.device)  # 用于生成数据
    Random_agent = RandomAgent(action_type, False)

    if os.path.exists(args.save_dir + "/checkpoint_ppo_v0.pt"):
        PPO_agent_learner.load(args.save_dir)
        PPO_agent_actor.load(args.save_dir)

    agents.append(PPO_agent_actor)
    for _ in range(env.n_player - 1):
        agents.append(Random_agent)
        # agents.append(PPO_agent_actor)

    new_trajectories = []
    reward = None
    writer = SummaryWriter(comment='-test_reward')
    while True:
        trajectories, payoff = run(env, agents)
        trajectories = reorganize(trajectories, payoff)[0]
        '''for player in range(env.n_player):
            for transition in trajectories[player]:
                if len(new_trajectories) < args.trajectory_size:
                    new_trajectories.append(transition)'''
        for transition in trajectories:
            if len(new_trajectories) < args.trajectory_size:
                new_trajectories.append(transition)
        print("trajectories length is %d" % len(new_trajectories))
        if len(new_trajectories) < args.trajectory_size:
            continue
        best_reward = PPO_agent_learner.train(new_trajectories)
        # PPO_agent_learner.train(new_trajectories)
        if PPO_agent_learner.total_steps % args.sync_steps == 0:
            '''for agent in agents:
                agent.actor_net.load_state_dict(PPO_agent_learner.actor_net.state_dict())'''
            agents[0].actor_net.load_state_dict(PPO_agent_learner.actor_net.state_dict())

        if PPO_agent_learner.total_steps % args.test_steps == 0:
            reward = test_net(env, PPO_agent_learner, Random_agent)
            writer.add_scalar('test_reward', reward, PPO_agent_learner.total_steps)

        '''if PPO_agent_learner.best_rewards is None or PPO_agent_learner.best_rewards < reward:
            PPO_agent_learner.save(args.save_dir)
            if PPO_agent_learner.best_rewards is not None:
                print("Best_mean_rewards updated %.2f -> %.2f, Model saved" % (PPO_agent_learner.best_rewards, reward))
            PPO_agent_learner.best_rewards = reward'''

        # print("best_rewards is %.2f" % best_reward)
        if best_reward >= args.goal_reward:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="chessandcard-mahjong_v3")
    parser.add_argument("--seed", type=int, default=42)

    save_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--save_dir", type=str, default=save_dir)
    parser.add_argument("--trajectory_size", type=int, default=2049)
    parser.add_argument('--goal_reward', type=float, default=0.2)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--sync_steps", type=int, default=3200)
    parser.add_argument("--test_steps", type=int, default=1920)
    args = parser.parse_args()
    train(args)

    '''obs = env.reset()[0]['obs']
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

    env = make(args.env)
    env.set_seed(args.seed)
    agents = []

    observation_shape = env.input_dimension['player_0']['observation'].shape
    action_type = env.joint_action_space[0][0]
    PPO_agent_learner = PPOAgent(observation_shape, action_type.n, args.save_dir, args.device)  # 用于学习
    PPO_agent_actor = PPOAgent(observation_shape, action_type.n, args.save_dir, args.device)  # 用于生成数据
    Random_agent = RandomAgent(action_type, False)
    # agents.append(PPO_agent)
    for _ in range(env.n_player):
        agents.append(PPO_agent_actor)
    trajectories, payoff = run(env, agents)
    new_trajectories = reorganize(trajectories, payoff)
    print(new_trajectories[0][0][0]['observation'].shape)
    obs = new_trajectories[0][0][0]['observation']
    obs_v = torch.FloatTensor(obs).unsqueeze(0)
    print(PPO_agent_learner.critic_net(obs_v))
    states, _, _, _ = PPO_agent_learner.unpack(new_trajectories[0])
    print(len(new_trajectories[0]), len(states))
    for _ in range(100):
        trajectories, payoff = run(env, agents)'''
