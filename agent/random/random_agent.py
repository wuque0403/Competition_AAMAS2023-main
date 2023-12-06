import numpy as np

class RandomAgent(object):
    def __init__(self, action_space_list_each, is_act_continuous):
        self.action_space_list_each = action_space_list_each
        self.is_act_continuous = is_act_continuous

    def step(self, state):
        each = []
        if self.is_act_continuous:
            each = self.action_space_list_each.sample()
        else:
            if self.action_space_list_each.__class__.__name__ == "Discrete":
                each = [0] * self.action_space_list_each.n
                action_index = list(np.where(state['action_mask'] == 1))[0]
                idx = np.random.choice(action_index)
                # idx = self.action_space_list_each.sample()
                each[idx] = 1
            elif self.action_space_list_each.__class__.__name__ == "MultiDiscreteParticle":
                each = []
                nvec = self.action_space_list_each.high - self.action_space_list_each.low + 1
                sample_indexes = self.action_space_list_each.sample()

                for i in range(len(nvec)):
                    dim = nvec[i]
                    new_action = [0] * dim
                    index = sample_indexes[i]
                    new_action[index] = 1
                    each.extend(new_action)
        return each

    def sample(self):
        player = []
        if self.is_act_continuous:
            for j in range(len(self.action_space_list_each)):
                each = self.action_space_list_each[j].sample()
                player.append(each)
        else:
            player = []
            for j in range(len(self.action_space_list_each)):
                # each = [0] * action_space_list_each[j]
                # idx = np.random.randint(action_space_list_each[j])
                if self.action_space_list_each[j].__class__.__name__ == "Discrete":
                    each = [0] * self.action_space_list_each[j].n
                    idx = self.action_space_list_each[j].sample()
                    each[idx] = 1
                    player.append(each)
                elif self.action_space_list_each[j].__class__.__name__ == "MultiDiscreteParticle":
                    each = []
                    nvec = self.action_space_list_each[j].high
                    sample_indexes = self.action_space_list_each[j].sample()

                    for i in range(len(nvec)):
                        dim = nvec[i] + 1
                        new_action = [0] * dim
                        index = sample_indexes[i]
                        new_action[index] = 1
                        each.extend(new_action)
                    player.append(each)
        return player
