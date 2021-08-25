import numpy as np
import time
import gym
import logging
import math
import torch

from qnet import DQN
from robust import Robust
# from cnn import CNN
from deeplearning import CNN


logging.basicConfig(level=logging.ERROR)

N_STATES = 16
ACTIONS = [0, 1, 2, 3]
EPSILON = 0.9
GAMMA = 0.9
MAX_EPISODES = 10
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
    # cnn = CNN()
    # cnn.load_state_dict(torch.load('test01.pkl'))

    best_score = 0
    acc_score = 0
    best_num = 0
    min_num = 1e10


    history = []

    for i_episode in range(MAX_EPISODES):
        print('Episode %d' % i_episode, end=' ')
        observation = env.reset()
        avg_step_time = 0
        avg_action_time = 0

        episode_history = []
        for t in range(10000):
            if t % 10 == 0 or t > 4000:
                env.render(fresh_time=FRESH_TIME)

            time_start = time.time()
            action = rl.choose_action(observation)    
            time_end = time.time()
            avg_action_time += time_end - time_start

            hist = {}
            hist['state'] = observation.copy()
            hist['action'] = action            

            episode_history.append(hist)

            time_start = time.time()
            observation_, reward, done, info = env.step(action)
            time_end = time.time()
            avg_step_time += time_end - time_start

            state_ = list(observation_.flat)

            observation = observation_

            if done:
                env.render()

                hist = {}
                hist['state'] = observation.copy()
                hist['action'] = action            

                episode_history.append(hist)

                acc_score += env.core.score
                if env.core.score > best_score:
                    best_score = env.core.score
                if max(env.core.board.flat) > best_num:
                    best_num = max(env.core.board.flat)
                if max(env.core.board.flat) < min_num:
                    min_num = max(env.core.board.flat)
                print('Score = %d, Best score = %d, Avg score = %d, Max number = % d, Min number = %d' \
                    % (env.core.score, best_score, acc_score / (i_episode + 1), best_num, min_num))
                print('avg step time = %f, avg action time = %f' % (avg_step_time / (t + 1), avg_action_time / (t + 1)))

                file_name = time.strftime('robust_%m%d_%H%M', time.localtime())
                
                file_name += '_%d_%d.npy' % (env.core.score, max(env.core.board.flat))
                print('file name = %s' % file_name)

                np.save(file_name, episode_history)

                time.sleep(5)
                break

    # replay
    history_load = np.load('robust_0821_1400_157556_8192.npy', allow_pickle=True)
    env.reset()
    input()
    replay_speed = 5
    for i in range(len(history_load)):
        state = history_load[i]['state']
        action = history_load[i]['action']
        # print('eps %d, action = %d' % (i - 20, action))
        env.core.board = state
        if i % replay_speed == 0:
            env.render(fresh_time=FRESH_TIME)

    input()
    env.close()