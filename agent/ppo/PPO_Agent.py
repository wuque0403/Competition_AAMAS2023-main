import torch as nn
import torch
import numpy as np
import torch.nn.functional as F
from collections import deque
import os
from tensorboardX import SummaryWriter

PPO_epoches = 10
batch_size = 32
trajectory_size = 2049
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPS = 0.2
learning_rate = 1e-5


class PPOAgent(object):
    def __init__(self, state_shape, num_actions, save_dir, device="cpu"):
        self.use_raw = False
        # self.actor_net_old = ActorNet(state_shape, num_actions)
        self.actor_net = ActorNet(state_shape, num_actions)
        self.critic_net = CriticNet(state_shape)
        self.optim_act = torch.optim.Adam(self.actor_net.parameters(), lr=learning_rate)
        self.optim_critic = torch.optim.Adam(self.critic_net.parameters(), lr=learning_rate)
        self.device = device
        self.save_dir = save_dir
        self.total_rewards = deque(maxlen=100)  # 记录最近100场对局的奖励
        self.best_rewards = None
        self.total_steps = 0
        self.writer = SummaryWriter(comment='-mahjong')
        # if os.path.exists(save_dir+"/checkpoint_ppo_v0.pth"):
            # self.load(save_dir+"/checkpoint_ppo_v0.pth")

    def step(self, state):
        obs = torch.FloatTensor(state['observation']).unsqueeze(0)
        action_index = list(np.where(state['action_mask'] == 1))[0]
        action_mask = list(state['action_mask'])
        policy_values = self.actor_net(obs).squeeze(0).detach()
        legal_actions_values = []

        for index, mask in enumerate(action_mask):
            if mask == 1:
                legal_actions_values.append(policy_values[index])

        best_action_index = action_index[np.argmax(legal_actions_values)]
        best_action = [0] * state['action_mask'].shape[0]
        best_action[best_action_index] = 1
        return best_action

    def train(self, trajectory):
        states, actions, rewards, dones = self.unpack(trajectory)
        trajectory_adv, trajectory_ref = self.calc_adv_ref(self.critic_net, states, rewards, dones, self.device)
        # obs_array = [state['observation'] for state in states]
        trajectory_obs = torch.FloatTensor(states).to(self.device)
        trajectory_actions = torch.Tensor(actions).long().to(self.device)
        old_policy_prob = torch.log(self.actor_net(trajectory_obs).gather(1,
                                    trajectory_actions.unsqueeze(-1)).squeeze(-1) + 1e-6).detach()
        trajectory_adv = (trajectory_adv - torch.mean(trajectory_adv)) / torch.std(trajectory_adv)
        trajectory_ref = (trajectory_ref - torch.mean(trajectory_ref)) / torch.std(trajectory_ref)

        array_dones = np.array(dones)
        episode_end_index = list(np.where(array_dones==True))[0]
        for index in episode_end_index:
            self.total_rewards.append(rewards[index])
        mean_episodes_reward = float(np.mean(self.total_rewards))
        print("one hundred games mean reward : %.2f" % mean_episodes_reward)
        self.writer.add_scalar('mean_episodes_reward', mean_episodes_reward, self.total_steps)

        for epoch in range(PPO_epoches):
            sum_loss_value = 0
            sum_policy_value = 0
            for batch_idx in range(0, trajectory_size-1, batch_size):
                self.total_steps += 1
                batch_obs = trajectory_obs[batch_idx:batch_idx+batch_size]
                batch_actions = trajectory_actions[batch_idx:batch_idx+batch_size]
                batch_adv = trajectory_adv[batch_idx:batch_idx+batch_size]
                batch_ref = trajectory_ref[batch_idx:batch_idx+batch_size]
                batch_old_prob = old_policy_prob[batch_idx:batch_idx+batch_size]

                # training critic_net
                self.optim_critic.zero_grad()
                value_v = self.critic_net(batch_obs)
                loss_value = F.mse_loss(value_v.squeeze(-1), batch_ref)
                loss_value.backward()
                self.optim_critic.step()

                # training actor_net
                self.optim_act.zero_grad()
                new_policy_prob = torch.log(self.actor_net(batch_obs).gather(1, batch_actions.unsqueeze(-1)).
                                            squeeze(-1) + 1e-6)
                ratio_v = torch.exp(new_policy_prob-batch_old_prob)
                surr_obj_v = ratio_v * batch_adv
                clipped_surr_obj_v = batch_adv * torch.clamp(ratio_v, 1.0-PPO_EPS, 1.0+PPO_EPS)
                loss_policy_v = -torch.min(surr_obj_v, clipped_surr_obj_v).mean()
                loss_policy_v.backward()
                self.optim_act.step()

                sum_loss_value += loss_value.item()
                sum_policy_value += loss_policy_v.item()
                # print("max_ratio: %.2f, max_adv: %.2f" % (torch.max(ratio_v).item(), torch.max(batch_adv).item()))
            print("Step%d——loss_value: %.3f, loss_policy_value: %.3f" % (self.total_steps, sum_loss_value,
                                                                         sum_policy_value))
            self.writer.add_scalar('sum_loss_value', sum_loss_value, self.total_steps)
            self.writer.add_scalar('sum_policy_loss', sum_policy_value, self.total_steps)
        trajectory.clear()

        if self.best_rewards is None or self.best_rewards < mean_episodes_reward:
            self.save(self.save_dir)
            if self.best_rewards is not None:
                print("Best_mean_rewards updated %.2f -> %.2f, Model saved" % (self.best_rewards, mean_episodes_reward))
            self.best_rewards = mean_episodes_reward
        print("best_rewards is %.2f" % self.best_rewards)
        return self.best_rewards

    def calc_adv_ref(self, critic_net, states, rewards, dones, device="cpu"):
        # obs_array = [state['observation'] for state in states]
        obs_v = torch.FloatTensor(states)
        values_v = critic_net(obs_v)
        values = values_v.squeeze().data.cpu().numpy()
        last_gae = 0.0
        result_adv = []
        result_ref = []
        for val, next_val, reward, done in zip(reversed(values[:-1]), reversed(values[1:]), reversed(rewards[:-1]),
                                               reversed(dones[:-1])):
            if done:
                delta = reward - val
                last_gae = delta
            else:
                delta = reward + GAMMA * next_val - val
                last_gae = delta + GAE_LAMBDA * GAMMA * last_gae
            result_adv.append(last_gae)
            result_ref.append(last_gae + val)
        result_adv = torch.FloatTensor(list(reversed(result_adv))).to(device)
        result_ref = torch.FloatTensor(list(reversed(result_ref))).to(device)
        return result_adv, result_ref

    def unpack(self, trajectory):
        states, actions, rewards, dones = [], [], [], []
        for i in range(len(trajectory)):
            states.append(trajectory[i][0]['observation'])
            actions.append(trajectory[i][1])
            rewards.append(trajectory[i][2])
            dones.append(trajectory[i][-1])
        return states, actions, rewards, dones

    def save(self, save_dir):
        checkpoint = {
            "actor_net": self.actor_net.state_dict(),
            "critic_net": self.critic_net.state_dict(),
            "actor_optimizer": self.optim_act.state_dict(),
            "critic_optimizer": self.optim_critic.state_dict()
        }
        torch.save(checkpoint, save_dir+"/checkpoint_ppo_v0.pt")

    def load(self, save_dir):
        checkpoint = torch.load(save_dir+"/checkpoint_ppo_v0.pt", encoding='UTF-8')
        # checkpoint = open(save_dir+"/checkpoint_ppo_v0.pt", encoding='UTF-8')
        self.actor_net.load_state_dict(checkpoint["actor_net"])
        self.critic_net.load_state_dict(checkpoint["critic_net"])
        self.optim_act.load_state_dict(checkpoint["actor_optimizer"])
        self.optim_critic.load_state_dict(checkpoint["critic_optimizer"])


class ActorNet(torch.nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ActorNet, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[0], 32, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=2, stride=2),
            torch.nn.ReLU(),
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(conv_out_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_actions),
            torch.nn.Softmax(dim=-1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class CriticNet(torch.nn.Module):
    def __init__(self, input_shape):
        super(CriticNet, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[0], 32, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=2, stride=2),
            torch.nn.ReLU(),
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(conv_out_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
            torch.nn.ReLU()
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
