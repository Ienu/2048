import numpy as np
import time
import gym
import logging
import math

from qnet import DQN
from robust import Robust
from cnn import CNN


logging.basicConfig(level=logging.ERROR)

N_STATES = 16
ACTIONS = [0, 1, 2, 3]
EPSILON = 0.9
GAMMA = 0.9
MAX_EPISODES = 0
FRESH_TIME = 0.0


if __name__ == '__main__':
    env = gym.make('Game2048-v0')

    # rl = DQN(
    #     n_states=N_STATES,
    #     n_actions=4,
    #     epsilon=EPSILON,
    #     gamma=GAMMA,
    # )
    rl = Robust()
    cnn = CNN()

    best_score = 0
    acc_score = 0
    best_num = 0
    min_num = 1e10


    history = []

    for i_episode in range(MAX_EPISODES):
        print('Episode %d' % i_episode, end=' ')
        observation = env.reset()
        avg_step_time = 0

        episode_history = []
        for t in range(10000):
            env.render(fresh_time=FRESH_TIME)

            # action = np.random.randint(0, 4)
            # print('observation = ', observation)
            # state = list(observation.flat)
            # print('state = ', state)
            # action = rl.choose_action(state)
            action = rl.choose_action(observation)

            hist = {}
            hist['state'] = observation.copy()
            hist['action'] = action            

            episode_history.append(hist)
            # print('episode history = ', episode_history)

            # action = cnn.choose_action(observation)
            # c = input()

            time_start = time.time()
            observation_, reward, done, info = env.step(action)
            time_end = time.time()
            avg_step_time += time_end - time_start
            # print('reward = ', reward)

            state_ = list(observation_.flat)
            # rl.learn(state, action, reward, done, state_)

            observation = observation_

            if done:
                env.render()
                # print("Episode finished after {} timesteps".format(t+1), end=' ')
                acc_score += env.core.score
                if env.core.score > best_score:
                    best_score = env.core.score
                if max(env.core.board.flat) > best_num:
                    best_num = max(env.core.board.flat)
                if max(env.core.board.flat) < min_num:
                    min_num = max(env.core.board.flat)
                print('Score = %d, Best score = %d, Avg score = %d, Max number = % d, Min number = %d' \
                    % (env.core.score, best_score, acc_score / (i_episode + 1), best_num, min_num), end=' ')
                print('avg step time = ', avg_step_time / (t + 1))

                file_name = time.strftime('robust_%m%d_%H%M.npy', time.localtime())
                print('file name = ', file_name)

                np.save(file_name, episode_history)

                time.sleep(1)
                break

    # replay
    history_load = np.load('robust_0815_1706.npy', allow_pickle=True)
    # print(history_load)
    input()
    env.reset()
    replay_speed = 5
    for i in range(len(history_load) // replay_speed):
        state = history_load[i * replay_speed]['state']
        action = history_load[i * replay_speed]['action']
        print('eps %d, action = %d' % (i * replay_speed, action))
        # print(state)
        env.core.board = state
        env.render(fresh_time=FRESH_TIME)
        # input()

    input()
    env.close()