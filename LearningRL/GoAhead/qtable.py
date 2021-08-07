import numpy as np


class QTable:
    def __init__(self, n_states=6, epsilon=0.9, gamma=0.9, alpha=0.1, actions=[0, 1]):
        self.q_table = np.zeros([n_states, 2])

        self.n_states = n_states
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.actions = actions


    def choose_action(self, state):
        state_actions = self.q_table[state, :]  # 选出这个 state 的所有 action 值
        if (np.random.uniform() > self.epsilon) or (state_actions.all() == 0):  # 非贪婪 or 或者这个 state 还没有探索过
            action = np.random.choice(self.actions)
        else:
            action = state_actions.argmax()    # 贪婪模式
        return action


    def learn(self, state, action, reward, done, next_state):
        q = self.q_table[state, action]
        if done:
            q_target = reward
        else:
            q_target = reward + self.gamma * max(self.q_table[next_state[0], :])   #  实际的(状态-行为)值 (回合没结束)

        self.q_table[state, action] += self.alpha * (q_target - q)