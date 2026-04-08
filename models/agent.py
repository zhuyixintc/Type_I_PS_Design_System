import numpy as np
import tensorflow as tf


class Agent:
    def __init__(self):
        self.q_net = self.dqn_model()
        self.target_q_net = self.dqn_model()

    @staticmethod
    def dqn_model():
        q_net = tf.keras.Sequential()
        q_net.add(tf.keras.layers.Dense(1024, input_dim=2049, activation='relu', kernel_initializer='he_uniform'))
        q_net.add(tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform'))
        q_net.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        q_net.add(tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform'))
        q_net.add(tf.keras.layers.Dense(1, activation='linear', kernel_initializer='he_uniform'))
        q_net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
        return q_net

    def get_action_epsilon(self, input_feature, epsilon):
        if np.random.uniform() < epsilon:
            return np.random.randint(0, len(input_feature))
        return self.get_action(input_feature)

    def get_action(self, input_feature):
        valid_actions_input = tf.convert_to_tensor(input_feature, dtype=tf.float32)
        action_q = self.q_net(valid_actions_input)
        action = np.argmax(action_q.numpy(), axis=0)
        action = int(action)
        return action

    def update_target_network(self):
        self.target_q_net.set_weights(self.q_net.get_weights())

    def train_net(self, batch):
        action_feature_batch, action_features_batch, reward_batch, done_batch = batch
        action_feature_batch = tf.convert_to_tensor(action_feature_batch, dtype=tf.float32)
        current_q = self.q_net(action_feature_batch)
        target_q = np.copy(current_q)

        max_next_q = []
        for i in range(batch[0].shape[0]):
            action_features_batch[i] = tf.convert_to_tensor(action_features_batch[i], dtype=tf.float32)
            next_q = self.target_q_net(action_features_batch[i])
            max_next_q.append(np.amax(next_q, axis=0))

        for i in range(batch[0].shape[0]):
            target_q[i] = reward_batch[i] if done_batch[i] else reward_batch[i] + 'your params'.gamma * max_next_q[i]

        result = self.q_net.fit(x=action_feature_batch, y=target_q, verbose=0)
        return result.history['loss']

    def save_model(self, episode_count):
        self.q_net.save()  # your save path

    def load_model(self, episode_count):
        self.q_net = tf.keras.models.load_model()  # your load path

