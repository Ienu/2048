import numpy as np
import pandas as pd
import time
import gym
import logging


logging.basicConfig(level=logging.INFO)

N_STATES = 6
ACTIONS = [0, 1]
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 100
FRESH_TIME = 0.1


class QLearning:
    def __init__(self):
        self.q_table = np.zeros([N_STATES, 2])


    def choose_action(self, state):
        state_actions = self.q_table[state, :]  # 选出这个 state 的所有 action 值
        if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):  # 非贪婪 or 或者这个 state 还没有探索过
            action = np.random.choice(ACTIONS)
        else:
            action = state_actions.argmax()    # 贪婪模式
        return action


    def learn(self, state, action, reward, done, next_state):
        q = self.q_table[state, action]
        if done:
            q_target = reward
        else:
            q_target = reward + GAMMA * max(self.q_table[next_state[0], :])   #  实际的(状态-行为)值 (回合没结束)

        self.q_table[state, action] += ALPHA * (q_target - q)


if __name__ == "__main__":
    env = gym.make('GoAhead-v0')

    rl = QLearning()

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