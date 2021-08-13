import numpy as np
import time
import gym
import logging
import math

from qnet import DQN
from greedy import Greedy


logging.basicConfig(level=logging.ERROR)

N_STATES = 16
ACTIONS = [0, 1, 2, 3]
EPSILON = 0.9
GAMMA = 0.9
MAX_EPISODES = 500
FRESH_TIME = 0.01


if __name__ == '__main__':
    env = gym.make('Game2048-v0')

    # rl = DQN(
    #     n_states=N_STATES,
    #     n_actions=4,
    #     epsilon=EPSILON,
    #     gamma=GAMMA,
    # )
    rl = Greedy()

    best_score = 0
    acc_score = 0
    best_num = 0
    min_num = 1e10
    for i_episode in range(MAX_EPISODES):
        print('Episode %d' % i_episode)
        observation = env.reset()
        for t in range(10000):
            env.render(fresh_time=FRESH_TIME)

            # action = np.random.randint(0, 4)
            # print('observation = ', observation)
            # state = list(observation.flat)
            # print('state = ', state)
            # action = rl.choose_action(state)
            action = rl.choose_action(observation)

            observation_, reward, done, info = env.step(action)
            # print('reward = ', reward)

            state_ = list(observation_.flat)
            # rl.learn(state, action, reward, done, state_)

            observation = observation_

            if done:
                env.render()
                print("Episode finished after {} timesteps".format(t+1))
                acc_score += env.core.score
                if env.core.score > best_score:
                    best_score = env.core.score
                if max(env.core.board.flat) > best_num:
                    best_num = max(env.core.board.flat)
                if max(env.core.board.flat) < min_num:
                    min_num = max(env.core.board.flat)
                print('Score = %d, Best score = %d, Avg score = %d, Max number = % d, Min number = %d' \
                    % (env.core.score, best_score, acc_score / (i_episode + 1), best_num, min_num))

                time.sleep(1)
                break
    env.close()