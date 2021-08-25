import numpy as np
from game2048 import Core
import copy
import logging


class Robust:
    def __init__(self):
        self.core = Core()
        self.active_down = False


    def max_reward(self, state, depth, alpha=-1e10, beta=1e10):
        '''
        估计当前状态通过新动作能够获得的最大reward
        beta为上界，如果reward > beta，也不会被上一层的min接受，只会选择beta
        又由于本层取max，因此返回的reward不会比该值小，可以直接剪枝
        '''
        c = Core()
        reward = np.zeros([4, 1])
        suc = np.zeros([4, 1])

        for i in range(4):
            if i == 3 and self.active_down == False:
                break

            c.board = copy.deepcopy(state)
            suc[i], reward[i] = c.action(i)

            if depth >= 1 and suc[i] == 1:
                reward[i] = self.min_reward(c.board, depth=depth - 1, alpha=alpha, beta=beta)
                if reward[i] > alpha:
                    alpha = reward[i]

            if reward[i] >= beta:
                return reward[i]

            reward[i] *= suc[i]

        max_reward = np.max(reward)
        return max_reward


    def min_reward(self, state, depth, alpha=-1e10, beta=1e10):
        '''
        估计当前状态增加新块后能够获得的最小reward,
        alpha为下界，如果回报小于alpha，由于上一层取max，至少取到alpha
        本层取min，不会取到比该值更大的值，因此可以剪枝
        '''
        empty_list = []
        for i in range(4):
            for j in range(4):
                if state[i, j] == 0:
                    empty_list.append([i, j])

        min_reward = 1e10
        for empty in empty_list:
            _state = copy.deepcopy(state)

            value = [2, 4]  # 可考虑先取4加快剪枝
            for v in value:
                _state[empty[0], empty[1]] = v

                max_reward = self.max_reward(_state, depth=depth, alpha=alpha, beta=beta)

                # 取得最小的reward
                if max_reward < min_reward:
                    min_reward = max_reward

                # 更新beta
                if max_reward < beta:
                    beta = max_reward

                # 如果reward <= alpha， 上一层的max只会取alpha，可以剪枝
                if max_reward <= alpha:
                    return max_reward

        if len(empty_list) == 0:
            min_reward = 0

        return min_reward


    def choose_action(self, state):
        c = Core()
        reward = np.zeros([4, 1])
        suc = np.zeros([4, 1])
        accum_reward = np.zeros([4, 1])

        value_max = np.max(state.flat)

        emptys = 0
        for i in range(16):
            if state.flat[i] == 0:
                emptys += 1

        self.active_down = True
        if emptys <= 2:
            self.active_down = True
        else:
            self.active_down = False

        th1, th2 = 256, 2048
        b1, b2, b3 = 9, 4, 0
        alpha, beta = -1e-10, 1e10

        for i in range(4):
            if i == 3 and self.active_down == False:
                break

            c.board = copy.deepcopy(state)
            suc[i], reward[i] = c.action(i)

            if emptys >= b1 and value_max <= th1:
                depth = 0
            elif emptys >= b2:
                depth = 1
            elif emptys >= b3:
                depth = 2
            else:
                depth = 3

            if i == 3 and depth < 2:
                depth = 2

            min_reward = self.min_reward(c.board, depth=depth, alpha=alpha, beta=beta)
            if min_reward > alpha:
                alpha = min_reward

            accum_reward[i] = suc[i] * min_reward

        action = np.argmax(accum_reward)
        
        if suc[action] == 0:
            if np.max(suc[0:2]) != 0:
                action = np.random.randint(0, 2)
            elif suc[2] != 0:
                action = 2
            else:
                action = 3

        return action