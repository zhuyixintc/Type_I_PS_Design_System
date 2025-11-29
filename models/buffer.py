import random
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self):
        self.experiences = deque(maxlen='your params'.buffer_max_len)

    def store_experience(self, action_feature, action_features, reward, done):
        self.experiences.append((action_feature, action_features, reward, done))

    def sample_experience_batch(self):
        batch_size = min(128, len(self.experiences))
        sampled_experience_batch = random.sample(self.experiences, batch_size)
        action_feature_batch, action_features_batch, reward_batch, done_batch = [], [], [], []

        for i in sampled_experience_batch:
            action_feature_batch.append(i[0])
            action_features_batch.append(i[1])
            reward_batch.append(i[2])
            done_batch.append(i[3])

        return np.array(action_feature_batch), np.array(action_features_batch), np.array(reward_batch), np.array(done_batch)

