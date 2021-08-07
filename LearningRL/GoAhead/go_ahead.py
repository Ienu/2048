import numpy as np
import time
import logging

import gym
from gym import spaces
from gym.utils import seeding


class GoAhead(gym.Env):
    def __init__(self):
        logging.debug('<INIT FUNC>')
        self.min_position = 0
        self.max_position = 5
        self.goal_position = self.max_position

        self.low = np.array(
            [self.min_position], dtype=np.integer
        )
        self.high = np.array(
            [self.max_position], dtype=np.integer
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            self.low, self.high, dtype=np.integer
        )

        self.seed()


    def step(self, action):
        logging.debug('<STEP FUNC>')
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position = self.state
        position += action * 2 - 1
        position = np.clip(position, self.min_position, self.max_position)

        done = bool(
            position == self.goal_position
        )

        reward = 0.0
        if done:
            reward = 1.0

        self.state = position
        return np.array(self.state), reward, done, {}


    def reset(self):
        logging.debug('<RESET FUNC>')
        self.state = np.array([self.np_random.randint(self.min_position, self.max_position)])
        self.state[0] = 0
        logging.info('state = %d' % self.state)
        return np.array(self.state)


    def render(self, mode='human', fresh_time=0.1):
        logging.debug('<RENDER FUNC>')
        env_list = ['-'] * self.max_position + ['T']   # '-----T' our environment
        env_list[self.state[0]] = 'o'
        interaction = ''.join(env_list)
        print(interaction)
        time.sleep(fresh_time)


    def close(self):
        logging.debug('<CLOSE FUNC>')


    def seed(self, seed=None):
        logging.debug('<SEED FUNC>')
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        