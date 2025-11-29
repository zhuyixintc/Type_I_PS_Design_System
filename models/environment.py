from models.molecular_generation import initialise_state, get_valid_trajectory
from models.predictor import predictor
from models.utils import remove_dp


class Environment(object):
    def __init__(self):
        self.state = None
        self.count = None
        self.max_count = None

    # initialise the env
    def reset(self):
        self.state = initialise_state()
        self.count = 0
        self.max_count = 'your params'.max_count
        return self.state

    # get all possible actions
    def action_space(self):
        action_space = get_valid_trajectory(self.state)
        return action_space

    def step(self, action):
        self.count += 1
        self.state = action

        # p1 p2 donate your properties
        if '*' in self.state:
            smi = remove_dp(self.state)
            p1, p2 = predictor(smi)
        else:
            smi = self.state
            p1, p2 = predictor(smi)

        done = bool(self.count >= self.max_count)

        # here you define your reward function

        return self.state, p1, p2,  # here return the reward, # return if the episode is done

