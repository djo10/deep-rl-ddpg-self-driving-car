import argparse
import time

import keras.backend as keras_backend
import numpy as np
import tensorflow as tf

from actor import ActorNetwork
from critic import CriticNetwork
from gym_torcs import TorcsEnv
from noise import OrnsteinUhlenbeckActionNoise
from replay_buffer import ReplayBuffer

np.random.seed(1337)


def play(train_indicator):
    buffer_size = 100000
    batch_size = 32
    gamma = 0.99    # discount factor
    tau = 0.001     # Target Network HyperParameters
    lra = 0.0001    # Learning rate for Actor
    lrc = 0.001     # Learning rate for Critic

    action_dim = 1  # Steering angle
    state_dim = 21  # num of sensors input

    episode_count = 2000
    max_steps = 100000
    step = 0
    ou_sigma = 0.3

    train_stat_file = "train_stat.txt"
    actor_weights_file = "actor.h5"
    critic_weights_file = "critic.h5"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf_session = tf.Session(config=config)

    keras_backend.set_session(tf_session)

    actor = ActorNetwork(tf_session=tf_session, state_size=state_dim, action_size=action_dim, hidden_units=(300, 600), tau=tau, lr=lra)
    critic = CriticNetwork(tf_session=tf_session, state_size=state_dim, action_size=action_dim, hidden_units=(300, 600), tau=tau, lr=lrc)
    buffer = ReplayBuffer(buffer_size)

    ou = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim), sigma=ou_sigma * np.ones(action_dim))

    # Torcs environment - throttle and gear change controlled by client
    env = TorcsEnv(vision=False, throttle=False, gear_change=False)

    try:
        actor.model.load_weights(actor_weights_file)
        critic.model.load_weights(critic_weights_file)
        actor.target_model.load_weights(actor_weights_file)
        critic.target_model.load_weights(critic_weights_file)
        print("Weights loaded successfully")
    except:
        print("Cannot load weights")

    for i in range(episode_count):
        print("Episode : %s Replay buffer %s" % (i, len(buffer)))

        if i % 3 == 0:
            ob = env.reset(relaunch=True)  # relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        # 21 len state dimensions - https://arxiv.org/abs/1304.1672
        state = np.hstack((ob.angle, ob.track, ob.trackPos))

        total_reward = 0.
        for j in range(max_steps):
            loss = 0

            action_predicted = actor.model.predict(state.reshape(1, state.shape[0])) + ou()  # predict and add noise

            observation, reward, done, info = env.step(action_predicted[0])

            state1 = np.hstack((observation.angle, observation.track, observation.trackPos))

            buffer.add((state, action_predicted[0], reward, state1, done))  # add replay buffer

            # batch update
            batch = buffer.get_batch(batch_size)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + gamma * target_q_values[k]

            if train_indicator:
                loss += critic.model.train_on_batch([states, actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.get_gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.train_target_model()
                critic.train_target_model()

            total_reward += reward
            state = state1

            print("Episode %s - Step %s - Action %s - Reward %s - Loss %s" % (i, step, action_predicted[0][0], reward, loss))

            step += 1
            if done:
                break

        if i % 3 == 0 and train_indicator:
            print("Saving weights...")
            actor.model.save_weights(actor_weights_file, overwrite=True)
            critic.model.save_weights(critic_weights_file, overwrite=True)

        tm = time.strftime("%Y-%m-%d %H:%M:%S")
        episode_stat = "%s -th Episode. %s total steps. Total reward: %s. Time %s" % (i, step, total_reward, tm)
        print(episode_stat)
        with open(train_stat_file, "a") as outfile:
            outfile.write(episode_stat+"\n")

    env.end()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", type=int, help="train indicator", default=0)
    args = parser.parse_args()
    play(args.train)
