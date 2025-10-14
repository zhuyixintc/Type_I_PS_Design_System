import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import tensorflow as tf
from models.agent import Agent
from models.buffer import ReplayBuffer
from models.environment import Environment
from models.utils import get_input_feature, remove_dp
import params_dqn


def train_dqn_model(max_episodes=params_dqn.n_episodes,
                    epsilon_start=params_dqn.epsilon_start,
                    max_count=params_dqn.max_count):
    # save data
    main_path = './checkpoints/dqn/add_' + str(max_count)
    
    log_dir = main_path + '/log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = tf.summary.create_file_writer(log_dir)
    with open(main_path + '/log/reward.txt', 'w') as f:
        f.write('episode' + '\t' + 'episode_reward' + '\t' + 'reward' + '\n')
    with open(main_path + '/log/loss.txt', 'w') as f:
        f.write('episode' + '\t' + 'loss' + '\n')

    model_dir = main_path + '/model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(main_path + '/log/mol_path.txt', 'w') as f:
        f.write('episode' + '\t' + 'start_frag' + '\t' + 'SMILES' + '\t' + 'terminated' + '\t' + 'reward' + '\t' + 't1' + '\t' + 'st' + '\n')

    all_smi = []

    # model
    agent = Agent()
    buffer = ReplayBuffer()
    env = Environment()

    for episode_count in range(max_episodes):
        print('episode:', episode_count)
        start_frag = env.reset()
        done = False
        reward = 0
        episode_reward = 0
        final_smi = None
        epsilon = epsilon_start * (params_dqn.epsilon_decay ** episode_count)

        while not done:
            # current state
            step_left = env.max_count - env.count
            valid_actions_current = env.action_space()

            input_feature = []
            for act in valid_actions_current:
                if '*' in act:
                    act = remove_dp(act)
                    input_feature.append(get_input_feature(act) + [step_left])
                else:
                    input_feature.append(get_input_feature(act) + [step_left])

            a = agent.get_action_epsilon(input_feature, epsilon)
            action = valid_actions_current[a]

            if '*' in action:
                act_smi = remove_dp(action)
                action_feature = get_input_feature(act_smi) + [step_left]
            else:
                action_feature = get_input_feature(action) + [step_left]

            next_state, t1, st, reward, done = env.step(action)

            # next state
            step_left = env.max_count - env.count
            valid_actions_next = env.action_space()

            action_features = []
            for act in valid_actions_next:
                if '*' in act:
                    act = remove_dp(act)
                    action_features.append(get_input_feature(act) + [step_left])
                else:
                    action_features.append(get_input_feature(act) + [step_left])

            buffer.store_experience(action_feature, action_features, reward, done)

            episode_reward += reward

            # molecule evaluation
            if '*' in next_state:
                final_smi = remove_dp(next_state)
            else:
                final_smi = next_state

            # save data
            with open(main_path + '/log/mol_path.txt', 'a') as f:
                f.write(str(episode_count) + '\t' +
                        str(start_frag) + '\t' +
                        str(final_smi) + '\t' +
                        str(done) + '\t' +
                        str(float(reward)) + '\t' +
                        str(float(t1)) + '\t' +
                        str(float(st)) + '\n')

        if reward > 0.75 and final_smi not in all_smi:
            all_smi.append(final_smi)

        print('reward:', reward,
              'all_smi:', len(all_smi)
              )

        with writer.as_default():
            tf.summary.scalar('reward', float(reward), step=episode_count)
            tf.summary.scalar('episode_reward', float(episode_reward), step=episode_count)

        with open(main_path + '/log/reward.txt', 'a') as f:
            f.write(str(episode_count) + '\t' + str(float(episode_reward)) + '\t' + str(float(reward)) + '\n')

        if len(buffer.experiences) >= params_dqn.buffer_warm_up:
            experience_batch = buffer.sample_experience_batch()
            loss = agent.train_net(experience_batch)
            
            with writer.as_default():
                tf.summary.scalar('loss', loss[0], step=episode_count)

            with open(main_path + '/log/loss.txt', 'a') as f:
                f.write(str(episode_count) + '\t' + str(loss[0]) + '\n')

            if episode_count % params_dqn.model_update_cycle == 0:
                agent.update_target_network()
                agent.save_model(episode_count)


if __name__ == "__main__":
    train_dqn_model()

