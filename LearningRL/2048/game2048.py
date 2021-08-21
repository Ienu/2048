import numpy as np
import time
import random
import logging
import math

import gym
from gym import spaces
from gym.utils import seeding

import tkinter as tk
from PIL import Image, ImageTk
import sys

from cnn import CNN

# model prectict control

class Core:
    def __init__(self):
        self.size = 4
        self.reset()

        # weight 2 Best
        self.weights = np.array([
            [2.0, 1.8, 1.8, 2.0],
            [1.6, 1.4, 1.4, 1.6],
            [1.2, 1.0, 1.0, 1.2],
            [0.8, 0.6, 0.6, 0.8]])
        # self.weights = np.array([
        #     [10.0, 8.9, 8.9, 10.0],
        #     [4.4, 4.1, 4.1, 4.4],
        #     [2.0, 1.7, 1.7, 2.0],
        #     [0.8, 0.5, 0.5, 0.8]])


    def reset(self):
        self.score = 0
        self.suc2048 = False
        self.board = np.zeros([self.size, self.size], dtype=int)
        self.emerge()
        self.emerge()


    def emerge(self):
        # number
        em_num = random.randint(1, 2) * 2

        # position
        empty_list = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == 0:
                    empty_list.append([i, j])

        if empty_list == []:
            return False

        ch = random.choice(empty_list)

        # set
        self.board[ch[0], ch[1]] = em_num
        return True


    def _test(self):
        # up test
        for col in range(self.size):
            lst = []
            for row in range(self.size):
                value = self.board[row, col]
                if value == 0:
                    return False
                else:
                    lst.append(self.board[row, col])

            for i in range(self.size - 1):
                if lst[i] == lst[i + 1]:
                    return False
        
        # down test
        for col in range(self.size):
            lst = []
            for row in range(self.size):
                value = self.board[self.size - 1 - row, col]
                if value == 0:
                    return False
                else:
                    lst.append(self.board[self.size - 1 - row, col])

            for i in range(self.size - 1):
                if lst[i] == lst[i + 1]:
                    return False

        # left test
        for row in range(self.size):
            lst = []
            for col in range(self.size):
                value = self.board[row, col]
                if value == 0:
                    return False
                else:
                    lst.append(self.board[row, col])

            for i in range(self.size - 1):
                if lst[i] == lst[i + 1]:
                    return False

        # left right
        for row in range(self.size):
            lst = []
            for col in range(self.size):
                value = self.board[row, self.size - 1 - col]
                if value == 0:
                    return False
                else:
                    lst.append(self.board[row, self.size - 1 - col])

            for i in range(self.size - 1):
                if lst[i] == lst[i + 1]:
                    return False
        
        return True


    def _plus(self, lst):
        res = []
        r = 0
        b_plus = False
        for v in lst:
            if res == []:
                res.append(v)
            else:
                if res[-1] == v and b_plus == False:
                    res[-1] += v
                    r = v * 2
                    self.score += r

                    if res[-1] == 2048:
                        self.suc2048 = True
                    b_plus = True
                else:
                    res.append(v)
                    b_plus = False
        return res, r


    def action(self, idx):
        if idx == 0:
            return self.action_up()
        elif idx == 1:
            return self.action_left()
        elif idx == 2:
            return self.action_right()
        elif idx == 3:
            return self.action_down()
        else:
            print('ACTION INDEX ERROR!')
            exit(0)



    def action_up(self):
        suc = False
        reward = 0
        for col in range(self.size):
            lst = []
            # push list
            for row in range(self.size):
                value = self.board[row, col]
                if value != 0:
                    lst.append(value)

            # plus
            res, r = self._plus(lst)
            reward += r
            
            # set
            row = 0
            for v in res:
                if self.board[row, col] != v:
                    suc = True

                self.board[row, col] = v
                row += 1

            for i in range(row, self.size):
                self.board[i, col] = 0

        ext_reward = 0
        for i in range(self.size):
            for j in range(self.size):
                ext_reward += self.board[i, j] * self.weights[i, j]

        return suc, ext_reward


    def action_down(self):
        suc = False
        reward = 0
        for col in range(self.size):
            lst = []
            # push list
            for row in range(self.size):
                value = self.board[self.size - 1 - row, col]
                if value != 0:
                    lst.append(value)

            # plus
            res, r = self._plus(lst)
            reward += r
            
            # set
            row = 0
            for v in res:
                if self.board[self.size - 1 - row, col] != v:
                    suc = True
                    
                self.board[self.size - 1 - row, col] = v
                row += 1

            for i in range(row, self.size):
                self.board[self.size - 1 - i, col] = 0

        ext_reward = 0
        for i in range(self.size):
            for j in range(self.size):
                ext_reward += self.board[i, j] * self.weights[i, j]

        return suc, ext_reward


    def action_left(self):
        suc = False
        reward = 0
        for row in range(self.size):
            lst = []
            # push list
            for col in range(self.size):
                value = self.board[row, col]
                if value != 0:
                    lst.append(value)

            # plus
            res, r = self._plus(lst)
            reward += r
            
            # set
            col = 0
            for v in res:
                if self.board[row, col] != v:
                    suc = True
                    
                self.board[row, col] = v
                col += 1

            for i in range(col, self.size):
                self.board[row, i] = 0

        ext_reward = 0
        for i in range(self.size):
            for j in range(self.size):
                ext_reward += self.board[i, j] * self.weights[i, j]

        return suc, ext_reward


    def action_right(self):
        suc = False
        reward = 0
        for row in range(self.size):
            lst = []
            # push list
            for col in range(self.size):
                value = self.board[row, self.size - 1 - col]
                if value != 0:
                    lst.append(value)

            # plus
            res, r = self._plus(lst)
            reward += r
            
            # set
            col = 0
            for v in res:
                if self.board[row, self.size - 1 - col] != v:
                    suc = True

                self.board[row, self.size - 1 - col] = v
                col += 1

            for i in range(col, self.size):
                self.board[row, self.size - 1 - i] = 0

        ext_reward = 0
        for i in range(self.size):
            for j in range(self.size):
                ext_reward += self.board[i, j] * self.weights[i, j]

        return suc, ext_reward