import numpy as np
import time
import gym
import logging

from qnet import DQN


logging.basicConfig(level=logging.ERROR)

N_STATES = 16
ACTIONS = [0, 1, 2, 3]
EPSILON = 0.9
GAMMA = 0.9
MAX_EPISODES = 500
FRESH_TIME = 0.01


if __name__ == '__main__':
    env = gym.make('Game2048-v0')

    rl = DQN(
        n_states=N_STATES,
        n_actions=4,
        epsilon=EPSILON,
        gamma=GAMMA,
    )

    for i_episode in range(MAX_EPISODES):
        print('Episode %d' % i_episode)
        observation = env.reset()
        for t in range(10000):
            env.render(fresh_time=FRESH_TIME)

            # action = np.random.randint(0, 4)
            # print('observation = ', observation)
            state = list(observation.flat)
            # print('state = ', state)
            action = rl.choose_action(state)

            observation_, reward, done, info = env.step(action)
            # print('reward = ', reward)

            state_ = list(observation_.flat)
            rl.learn(state, action, reward, done, state_)

            observation = observation_

            if done:
                env.render()
                print("Episode finished after {} timesteps".format(t+1))


                time.sleep(1)
                break
    env.close()