#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Copyright (c). All Rights Reserved.
# -----------------------------------------------------
# File Name:        core.py
# Creator:          Wenyu Li
# Version:          0.2
# Created:          2021/08/02
# Description:      core program for 2048 game
# Function List:    class Core
# History:
#   <author>      <version>       <time>          <description>
#   Wenyu Li      0.1             2021/08/02       create
#   Wenyu Li      0.2             2021/08/04       add size attribute
# -----------------------------------------------------

import numpy as np
import random


class Core:
    def __init__(self):
        self.size = 2
        self.reset()


    def reset(self):
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

                    if res[-1] == 16:
                        self.suc2048 = True
                    b_plus = True
                else:
                    res.append(v)
                    b_plus = False
        return res, r


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

        # print('up reward = %d' % reward)
        return suc, reward


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

        # print('down reward = %d' % reward)
        return suc, reward


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

        # print('left reward = %d' % reward)
        return suc, reward


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

        # print('right reward = %d' % reward)
        return suc, reward


    def deb_show(self):
        for i in range(self.size):
            for j in range(self.size):
                print(self.board[i, j], end='  ')
            print('')


if __name__ == '__main__':
    c = Core()
    c.deb_show()