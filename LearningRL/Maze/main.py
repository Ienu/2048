import numpy as np
import time
import gym
import logging

from qtable import QTable
from qnet import DQN


logging.basicConfig(level=logging.ERROR)

N_STATES = 16
ACTIONS = [0, 1, 2, 3]
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 500
FRESH_TIME = 0.1


if __name__ == "__main__":
    env = gym.make('Maze-v0')

    rl1 = QTable(
        n_states=N_STATES,
        epsilon=EPSILON,
        gamma=GAMMA,
        alpha=ALPHA,
        actions=ACTIONS
        )
    
    rl2 = DQN(
        n_states=2,
        n_actions=4,
        epsilon=EPSILON,
        gamma=GAMMA
    )

    rl = rl2

    for i_episode in range(MAX_EPISODES):
        print('Episode %d' % i_episode)
        observation = env.reset()
        for t in range(100):
            env.render(fresh_time=FRESH_TIME)

            # action = np.random.randint(0, 4)
            # state = [int((observation[0] * 4 + 2) * 4 + observation[1] * 4 + 2)]
            state = [int(observation[0] * 4), int(observation[1] * 4)]
            action = rl.choose_action(state)

            observation_, reward, done, info = env.step(action)
            state_ = [int(observation_[0] * 4), int(observation_[1] * 4)]
            rl.learn(state, action, reward, done, state_)

            observation = observation_

            if done:
                env.render()
                print("Episode finished after {} timesteps".format(t+1))
                print('\r\nQ-table:\n')
                table = rl.show_table()
                env.show_prob(table)
                time.sleep(1)
                break

    env.close()