import numpy as np
import pandas as pd
import time
import gym
import logging

from qtable import QTable


logging.basicConfig(level=logging.INFO)

N_STATES = 6
ACTIONS = [0, 1]
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 100
FRESH_TIME = 0.1


if __name__ == "__main__":
    env = gym.make('GoAhead-v0')

    rl = QTable(
        n_states=N_STATES,
        epsilon=EPSILON,
        gamma=GAMMA,
        alpha=ALPHA,
        actions=ACTIONS
        )

    for i_episode in range(MAX_EPISODES):
        observation = env.reset()
        for t in range(100):
            env.render(fresh_time=FRESH_TIME)
            action = rl.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            rl.learn(observation, action, reward, done, observation_)
            observation = observation_

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                print('\r\nQ-table:\n')
                print(rl.q_table)
                time.sleep(1)
                break

    env.close()