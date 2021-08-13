import numpy as np
from game2048 import Core
import copy


class Greedy:
    def __init__(self):
        self.core = Core()
        self.gamma = 1


    def _max_reward(self, state):
        '''
        估计当前状态通过新动作能够获得的最大reward
        '''
        c = Core()
        reward = np.zeros([4, 1])
        suc = np.zeros([4, 1])

        c.board = copy.deepcopy(state)
        suc[0], reward[0] = c.action_up()

        c.board = copy.deepcopy(state)
        suc[1], reward[1] = c.action_left()

        c.board = copy.deepcopy(state)
        suc[2], reward[2] = c.action_right()

        # c.board = copy.deepcopy(state)
        # suc[3], reward[3] = c.action_down()
        for i in range(4):
            reward[i] *= suc[i]

        max_reward = np.max(reward)
        return max_reward


    def _min_reward(self, state):
        '''
        估计当前状态增加新块后能够获得的最小reward
        '''
        # c = Core()
        # c.board = copy.deepcopy(state)

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
                max_reward = self._max_reward(state)
                if max_reward < min_reward:
                    min_reward = max_reward
        
        return min_reward


    def _value_max(self, state):
        c = Core()
        reward = np.zeros([4, 1])
        suc = np.zeros([4, 1])
        accum_reward = np.zeros([4, 1])

        c.board = copy.deepcopy(state)
        suc[0], reward[0] = c.action_up()
        min_reward = self._min_reward(c.board)
        accum_reward[0] = self.gamma * min_reward + reward[0]

        c.board = copy.deepcopy(state)
        suc[1], reward[1] = c.action_left()
        min_reward = self._min_reward(c.board)
        accum_reward[1] = self.gamma * min_reward + reward[1]

        c.board = copy.deepcopy(state)
        suc[2], reward[2] = c.action_right()
        min_reward = self._min_reward(c.board)
        accum_reward[2] = self.gamma * min_reward + reward[2]

        for i in range(4):
            accum_reward[i] *= suc[i]

        value = np.max(accum_reward)
        return value


    def _value_min(self, state):
        # c = Core()
        # c.board = copy.deepcopy(state)

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
                max_reward = self._value_max(_state)
                if max_reward < min_reward:
                    min_reward = max_reward
        
        return min_reward


    def choose_action(self, state):
        c = Core()
        reward = np.zeros([4, 1])
        suc = np.zeros([4, 1])
        accum_reward = np.zeros([4, 1])

        c.board = copy.deepcopy(state)
        suc[0], reward[0] = c.action_up()
        min_reward = self._value_min(c.board)
        # min_reward = self._min_reward(c.board)
        accum_reward[0] = self.gamma * min_reward + reward[0]

        c.board = copy.deepcopy(state)
        suc[1], reward[1] = c.action_left()
        min_reward = self._value_min(c.board)
        # min_reward = self._min_reward(c.board)
        accum_reward[1] = self.gamma * min_reward + reward[1]

        c.board = copy.deepcopy(state)
        suc[2], reward[2] = c.action_right()
        min_reward = self._value_min(c.board)
        # min_reward = self._min_reward(c.board)
        accum_reward[2] = self.gamma * min_reward + reward[2]

        # c.board = copy.deepcopy(state)
        # suc[3], reward[3] = c.action_down()
        # min_reward = self._min_reward(c.board)
        # accum_reward[3] = min_reward + reward[3]

        for i in range(4):
            accum_reward[i] *= suc[i]

        action = np.argmax(accum_reward)
        if suc[action] == 0:
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



        

