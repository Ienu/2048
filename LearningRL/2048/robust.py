import numpy as np
from game2048 import Core
import copy
from cnn import CNN


class Robust:
    def __init__(self):
        self.core = Core()
        self.cnn = CNN()
        self.gamma = 1
        self.active_down = False
        # self.weights = np.array([
        #     [1.5, 1.2, 1.2, 1.5],
        #     [1.2, 1.0, 1.0, 1.2],
        #     [1.0, 0.8, 0.8, 1.0],
        #     [0.8, 0.6, 0.6, 0.8]])


    def _max_reward(self, state, alpha=-1e10, beta=1e10):
        '''
        估计当前状态通过新动作能够获得的最大reward
        beta为上界，如果reward > beta，也不会被上一层的min接受，只会选择beta
        又由于本层取max，因此返回的reward不会比该值小，可以直接剪枝
        '''
        c = Core()
        reward = np.zeros([4, 1])
        suc = np.zeros([4, 1])

        c.board = copy.deepcopy(state)
        suc[0], reward[0] = c.action_up()
        if reward[0] >= beta:
            return reward[0]

        c.board = copy.deepcopy(state)
        suc[1], reward[1] = c.action_left()
        if reward[1] >= beta:
            return reward[1]

        c.board = copy.deepcopy(state)
        suc[2], reward[2] = c.action_right()
        if reward[2] >= beta:
            return reward[2]

        if self.active_down:
            c.board = copy.deepcopy(state)
            suc[3], reward[3] = c.action_down()
            if reward[3] >= beta:
                return reward[3]

        for i in range(4):
            reward[i] *= suc[i]

        max_reward = np.max(reward)

        return max_reward


    def _min_reward(self, state, alpha=-1e10, beta=1e10):
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

            value = [2, 4]
            for v in value:
                _state[empty[0], empty[1]] = v

                reward = self._max_reward(state, alpha, beta)

                # 取得最小的reward
                if reward < min_reward:
                    min_reward = reward

                # 更新beta
                if reward < beta:
                    beta = reward

                # 如果reward <= alpha， 上一层的max只会取alpha，可以剪枝
                if reward <= alpha:
                    return reward

        return min_reward


    def _value_max(self, state, alpha=-1e10, beta=1e10):
        c = Core()
        reward = np.zeros([4, 1])
        suc = np.zeros([4, 1])
        accum_reward = np.zeros([4, 1])

        c.board = copy.deepcopy(state)
        suc[0], reward[0] = c.action_up() 
        accum_reward[0] = self._min_reward(c.board, alpha, beta)

        if accum_reward[0] > alpha:
            alpha = accum_reward[0]

        if accum_reward[0] >= beta:
            return accum_reward[0]

        c.board = copy.deepcopy(state)
        suc[1], reward[1] = c.action_left()
        accum_reward[1] = self._min_reward(c.board, alpha, beta)

        if accum_reward[1] > alpha:
            alpha = accum_reward[1]

        if accum_reward[1] >= beta:
            return accum_reward[1]

        c.board = copy.deepcopy(state)
        suc[2], reward[2] = c.action_right()
        accum_reward[2] = self._min_reward(c.board, alpha, beta)

        if accum_reward[2] > alpha:
            alpha = accum_reward[2]

        if accum_reward[2] >= beta:
            return accum_reward[2]

        if self.active_down:
            c.board = copy.deepcopy(state)
            suc[3], reward[3] = c.action_down()
            accum_reward[3] = self._min_reward(c.board, alpha, beta)

            if accum_reward[3] > alpha:
                alpha = accum_reward[3]

            if accum_reward[3] >= beta:
                return accum_reward[3]

        for i in range(4):
            accum_reward[i] *= suc[i]

        value = np.max(accum_reward)

        return value


    def _value_min(self, state, alpha=-1e10, beta=1e10):

        empty_list = []
        for i in range(4):
            for j in range(4):
                if state[i, j] == 0:
                    empty_list.append([i, j])

        # min_reward = 1e10
        for empty in empty_list:
            _state = copy.deepcopy(state)

            value = [2, 4]
            for v in value:
                _state[empty[0], empty[1]] = v
                max_reward = self._value_max(_state, alpha, beta)

                # if max_reward < min_reward:
                #     min_reward = max_reward

                if max_reward < beta:
                    beta = max_reward

                if max_reward <= alpha:
                    return max_reward
        
        return beta


    def _value_max2(self, state, alpha=-1e10, beta=1e10):
        c = Core()
        reward = np.zeros([4, 1])
        suc = np.zeros([4, 1])
        accum_reward = np.zeros([4, 1])

        c.board = copy.deepcopy(state)
        suc[0], reward[0] = c.action_up()
        accum_reward[0] = self._value_min(c.board, alpha, beta)

        if accum_reward[0] > alpha:
            alpha = accum_reward[0]

        if accum_reward[0] >= beta:
            return accum_reward[0]

        c.board = copy.deepcopy(state)
        suc[1], reward[1] = c.action_left()
        accum_reward[1] = self._value_min(c.board, alpha, beta)

        if accum_reward[1] > alpha:
            alpha = accum_reward[1]

        if accum_reward[1] >= beta:
            return accum_reward[1]

        c.board = copy.deepcopy(state)
        suc[2], reward[2] = c.action_right()
        accum_reward[2] = self._value_min(c.board, alpha, beta)

        if accum_reward[2] > alpha:
            alpha = accum_reward[2]

        if accum_reward[2] >= beta:
            return accum_reward[2]

        if self.active_down:
            c.board = copy.deepcopy(state)
            suc[3], reward[3] = c.action_down()
            accum_reward[3] = self._value_min(c.board, alpha, beta)

            if accum_reward[3] > alpha:
                alpha = accum_reward[3]

            if accum_reward[3] >= beta:
                return accum_reward[3]

        for i in range(4):
            accum_reward[i] *= suc[i]

        value = np.max(accum_reward)
        return value


    def _value_min2(self, state, alpha=-1e10, beta=1e10):

        empty_list = []
        for i in range(4):
            for j in range(4):
                if state[i, j] == 0:
                    empty_list.append([i, j])

        # min_reward = 1e10
        for empty in empty_list:
            _state = copy.deepcopy(state)

            value = [2, 4]
            for v in value:
                _state[empty[0], empty[1]] = v
                max_reward = self._value_max2(_state, alpha, beta)
                # if max_reward < min_reward:
                #     min_reward = max_reward

                if max_reward < beta:
                    beta = max_reward

                if max_reward <= alpha:
                    return max_reward
        
        return beta


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
        # print('value max = ', value_max)

        # self.active_down = True
        # if emptys <= 2:
        #     self.active_down = True
        # else:
        #     self.active_down = False

        th1, th2 = 256, 2048
        b1, b2 = 8, 3
        alpha, beta = -1e-10, 1e10


        c.board = copy.deepcopy(state)
        suc[0], reward[0] = c.action_up()

        if emptys >= b1 and value_max <= th1:
            min_reward = self._min_reward(c.board, alpha, beta)
            if min_reward > alpha:
                alpha = min_reward

        elif emptys >= b2:
            min_reward = self._value_min(c.board)
            if min_reward > alpha:
                alpha = min_reward

        else:
            min_reward = self._value_min2(c.board)
            if min_reward > alpha:
                alpha = min_reward

        accum_reward[0] = self.gamma * min_reward #+ reward[0]# - self.cnn._eval(c.board)

        c.board = copy.deepcopy(state)
        suc[1], reward[1] = c.action_left()
        # min_reward = self._value_min2(c.board)
        
        if emptys >= b1 and value_max <= th1:
            min_reward = self._min_reward(c.board, alpha, beta)
            if min_reward > alpha:
                alpha = min_reward

        elif emptys >= b2:
            min_reward = self._value_min(c.board)
            if min_reward > alpha:
                alpha = min_reward

        else:
            min_reward = self._value_min2(c.board)
            if min_reward > alpha:
                alpha = min_reward

        accum_reward[1] = self.gamma * min_reward #+ reward[1]# - self.cnn._eval(c.board)

        c.board = copy.deepcopy(state)
        suc[2], reward[2] = c.action_right()
        # min_reward = self._value_min2(c.board)
        
        if emptys >= b1 and value_max <= th1:
            min_reward = self._min_reward(c.board, alpha, beta)
            if min_reward > alpha:
                alpha = min_reward

        elif emptys >= b2:
            min_reward = self._value_min(c.board)
            if min_reward > alpha:
                alpha = min_reward

        else:
            min_reward = self._value_min2(c.board)
            if min_reward > alpha:
                alpha = min_reward

        accum_reward[2] = self.gamma * min_reward #+ reward[2]# - self.cnn._eval(c.board)

        if self.active_down:
            c.board = copy.deepcopy(state)
            suc[3], reward[3] = c.action_down()
            accum_reward[3] = self._value_min2(c.board)

        for i in range(4):
            accum_reward[i] *= suc[i]

        action = np.argmax(accum_reward)
        if suc[action] == 0 and self.active_down == False:
            if np.max(suc[0:2]) != 0:
                action = np.random.randint(0, 2)
            elif suc[2] != 0:
                action = 2
            else:
                action = 3
            # if np.max(suc[0:3]) != 0:
            #     action = np.random.randint(0, 3)
            # else:
            #     action = 3

        return action



        

