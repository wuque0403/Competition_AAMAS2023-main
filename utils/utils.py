def run(env, agents):
    trajectories = [[] for _ in range(env.n_player)]
    all_observes = env.reset()
    player_id = env.player_id_map[env.env_core.agent_selection]
    state = all_observes[player_id]['obs']
    trajectories[player_id].append(state)
    while not env.is_terminal():
        # print(env.env_core.agent_selection, player_id)
        action = agents[player_id].step(state)  # 形式是[0,0,..,1,0,0,...]
        decode_action = action.index(1)
        joint_action = []
        for i in range(env.n_player):
            if i == player_id:
                joint_action.append([action])
            else:
                joint_action.append([])
        all_observes, reward, done, _, _ = env.step(joint_action)
        trajectories[player_id].append(decode_action)
        player_id = env.player_id_map[env.env_core.agent_selection]
        state = all_observes[player_id]['obs']
        if not done:
            trajectories[player_id].append(state)
    # trajectories[player_id].append(state)
    payoff = env.payoff
    return trajectories, payoff


def reorganize(trajectories, payoffs):
    num_players = len(trajectories)
    new_trajectories = [[] for _ in range(num_players)]

    for player in range(num_players):
        for i in range(0, len(trajectories[player]) - 1, 2):
            if i == len(trajectories[player]) - 2:
                reward = payoffs['player_' + str(player)]
                done = True
            else:
                reward, done = 0, False
            transition = trajectories[player][i:i + 2].copy()
            transition.append(reward)
            transition.append(done)

            new_trajectories[player].append(transition)
    return new_trajectories
