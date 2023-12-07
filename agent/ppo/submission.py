import sys
from pathlib import Path
import json
import env
import os


def make(env_type, conf=None, seed=None):
    file_path = os.path.join(os.path.dirname(__file__), 'config.json')
    if not conf:
        with open(file_path) as f:
            conf = json.load(f)[env_type]
    class_literal = conf['class_literal']
    return getattr(env, class_literal)(conf)


base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))

# from env.chooseenv import make
ENV_NAME = "chessandcard-mahjong_v3"
environment = make(ENV_NAME)

observation_shape = environment.input_dimension['player_0']['observation'].shape
action_type = environment.joint_action_space[0][0]
save_dir = os.path.dirname(os.path.abspath(__file__))

from PPO_Agent import PPOAgent
PPO_agent = PPOAgent(observation_shape, action_type.n, save_dir)
PPO_agent.load(save_dir)
sys.path.pop(-1)


def my_controller(observation, action_space, is_act_continuous=False):
    agent_action = []
    player_id = int(observation['current_move_player'][-1])
    control_player = observation['controlled_player_index']
    for i in range(len(action_space)):
        if i == player_id and player_id == control_player:
            state = observation['obs']
            best_action = PPO_agent.step(state)
            agent_action.append(best_action)
        else:
            action = sample_single_dim(action_space[i], is_act_continuous)
            agent_action.append(action)
    return agent_action


def sample_single_dim(action_space_list_each, is_act_continuous):
    each = []
    if is_act_continuous:
        each = action_space_list_each.sample()
    else:
        if action_space_list_each.__class__.__name__ == "Discrete":
            each = [0] * action_space_list_each.n
            # action_idx = list(np.where(observation['obs']['action_mask'] == 1))[0]
            # idx = np.random.choice(action_idx)
            idx = action_space_list_each.sample()
            each[idx] = 1
        elif action_space_list_each.__class__.__name__ == "MultiDiscreteParticle":
            each = []
            nvec = action_space_list_each.high - action_space_list_each.low + 1
            sample_indexes = action_space_list_each.sample()

            for i in range(len(nvec)):
                dim = nvec[i]
                new_action = [0] * dim
                index = sample_indexes[i]
                new_action[index] = 1
                each.extend(new_action)
    return each
