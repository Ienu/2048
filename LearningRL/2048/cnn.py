import numpy as np
# from game2048 import Core
import copy



class CNN:
    def __init__(self):
        pass


    def choose_action(self, state):
        c = Core()
        reward = np.zeros([4, 1])
        suc = np.zeros([4, 1])
        value = np.zeros([4, 1])

        c.board = copy.deepcopy(state)
        suc[0], reward[0] = c.action_up()
        value[0] = self._eval(c.board) * suc[0]
        print('Value Up = ', value[0], end=' ')


        c.board = copy.deepcopy(state)
        suc[1], reward[1] = c.action_left()
        value[1] = self._eval(c.board) * suc[1]
        print('Value Left = ', value[1], end=' ')


        c.board = copy.deepcopy(state)
        suc[2], reward[2] = c.action_right()
        value[2] = self._eval(c.board) * suc[2]
        print('Value Right = ', value[2])

        min_value = 1e10
        min_action = -1
        for i in range(3):
            if value[i] != 0:
                if min_value > value[i]:
                    min_value = value[i]
                    min_action = i
        if min_action == -1:
            min_action = 3

        return min_action



    def _eval(self, state):
        _state = copy.deepcopy(state)
        # set all zeros to one
        for i in range(4):
            for j in range(4):
                if _state[i, j] == 0:
                    _state[i, j] = 1
            #     print(_state[i, j], end=' ')
            # print('')

        # cal adjacent ratio
        delta = np.array([[-1, 0, 1, 0], [0, -1, 0, 1]])
        # print('delta = ', delta)

        sum_ratio = 0
        for i in range(4):
            for j in range(4):
                value = _state[i, j]
                min_ratio = 1e10
                for k in range(4):
                    ni = i + delta[0, k]
                    nj = j + delta[1, k]
                    if ni >= 0 and ni < 4 and nj >= 0 and nj < 4:
                        adj_value = _state[ni, nj]
                        ratio = max(adj_value/value, value/adj_value)
                        if ratio < min_ratio:
                            min_ratio = ratio
                sum_ratio += min_ratio
        #         print(int(min_ratio), end=' ')
        #     print('')
        # print('sum_ratio = ', sum_ratio)
        return sum_ratio / 4

        