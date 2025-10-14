import numpy as np
from models.molecular_generation import initialise_state, get_valid_trajectory
from models.predictor import predictor
from models.utils import remove_dp
import params_dqn


class Environment(object):
    def __init__(self):
        self.state = None
        self.count = None
        self.max_count = None

    # initialise the env
    def reset(self):
        self.state = initialise_state()
        self.count = 0
        self.max_count = params_dqn.max_count
        return self.state

    # get all possible actions
    def action_space(self):
        action_space = get_valid_trajectory(self.state)
        return action_space

    def step(self, action):
        self.count += 1
        self.state = action

        if '*' in self.state:
            smi = remove_dp(self.state)
            t1, st = predictor(smi)
        else:
            smi = self.state
            t1, st = predictor(smi)

        done = bool(self.count >= self.max_count)

        if 0 < t1 < 0.977:
            reward_t1 = 1
        else:
            reward_t1 = 0

        if st > 0:
            reward_st = np.exp(-st)
        else:
            reward_st = 0

        reward = reward_t1 * params_dqn.reward_weight_t1 + reward_st * params_dqn.reward_weight_st

        if self.max_count == 1:
            reward = reward
        else:
            reward = reward * params_dqn.discount_factor ** (self.max_count - self.count)

        return self.state, t1, st, reward, done

