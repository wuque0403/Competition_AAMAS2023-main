import sys
from pathlib import Path
import os

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))

from env.chooseenv import make
ENV_NAME = "chessandcard-mahjong_v3"
env = make(ENV_NAME)

observation_shape = env.input_dimension['player_0']['observation'].shape
action_type = env.joint_action_space[0][0]
save_dir = os.path.dirname(os.path.abspath(__file__))

from PPO_Agent import PPOAgent
PPO_agent = PPOAgent(observation_shape, action_type.n, save_dir)
PPO_agent.load(save_dir)
sys.path.pop(-1)


def my_controller(observation, action_space, is_act_continuous=False):
    agent_action = []
    player_id = int(observation['current_move_player'][-1])
    for i in range(len(action_space)):
        if i == player_id:
            state = observation['obs']
            best_action = PPO_agent.step(state)
            agent_action.append(best_action)
        else:
            agent_action.append([])
    return agent_action
