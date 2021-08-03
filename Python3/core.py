#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Copyright (c). All Rights Reserved.
# -----------------------------------------------------
# File Name:        core.py
# Creator:          Wenyu Li
# Version:          0.1
# Created:          2021/08/02
# Description:      core program for 2048 game
# Function List:    class Core
# History:
#   <author>      <version>       <time>          <description>
#   Wenyu Li      0.1             2021/08/02       create
# -----------------------------------------------------

import numpy as np
import random


class Core:
    def __init__(self):
        self.reset()


    def reset(self):
        self.suc2048 = False
        self.board = np.zeros([4, 4], dtype=int)
        self.emerge()
        self.emerge()


    def emerge(self):
        # number
        em_num = random.randint(1, 2) * 2

        # position
        empty_list = []
        for i in range(4):
            for j in range(4):
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
        for col in range(4):
            lst = []
            for row in range(4):
                value = self.board[row, col]
                if value == 0:
                    return False
                else:
                    lst.append(self.board[row, col])

            for i in range(3):
                if lst[i] == lst[i + 1]:
                    return False
        
        # down test
        for col in range(4):
            lst = []
            for row in range(4):
                value = self.board[3 - row, col]
                if value == 0:
                    return False
                else:
                    lst.append(self.board[3 - row, col])

            for i in range(3):
                if lst[i] == lst[i + 1]:
                    return False

        # left test
        for row in range(4):
            lst = []
            for col in range(4):
                value = self.board[row, col]
                if value == 0:
                    return False
                else:
                    lst.append(self.board[row, col])

            for i in range(3):
                if lst[i] == lst[i + 1]:
                    return False

        # left right
        for row in range(4):
            lst = []
            for col in range(4):
                value = self.board[row, 3 - col]
                if value == 0:
                    return False
                else:
                    lst.append(self.board[row, 3 - col])

            for i in range(3):
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

                    if res[-1] == 128:
                        self.suc2048 = True
                    b_plus = True
                else:
                    res.append(v)
                    b_plus = False
        return res, r


    def action_up(self):
        suc = False
        reward = 0
        for col in range(4):
            lst = []
            # push list
            for row in range(4):
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

            for i in range(row, 4):
                self.board[i, col] = 0

        # print('up reward = %d' % reward)
        return suc, reward


    def action_down(self):
        suc = False
        reward = 0
        for col in range(4):
            lst = []
            # push list
            for row in range(4):
                value = self.board[3 - row, col]
                if value != 0:
                    lst.append(value)

            # plus
            res, r = self._plus(lst)
            reward += r
            
            # set
            row = 0
            for v in res:
                if self.board[3 - row, col] != v:
                    suc = True
                    
                self.board[3 - row, col] = v
                row += 1

            for i in range(row, 4):
                self.board[3 - i, col] = 0

        # print('down reward = %d' % reward)
        return suc, reward


    def action_left(self):
        suc = False
        reward = 0
        for row in range(4):
            lst = []
            # push list
            for col in range(4):
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

            for i in range(col, 4):
                self.board[row, i] = 0

        # print('left reward = %d' % reward)
        return suc, reward


    def action_right(self):
        suc = False
        reward = 0
        for row in range(4):
            lst = []
            # push list
            for col in range(4):
                value = self.board[row, 3 - col]
                if value != 0:
                    lst.append(value)

            # plus
            res, r = self._plus(lst)
            reward += r
            
            # set
            col = 0
            for v in res:
                if self.board[row, 3 - col] != v:
                    suc = True

                self.board[row, 3 - col] = v
                col += 1

            for i in range(col, 4):
                self.board[row, 3 - i] = 0

        # print('right reward = %d' % reward)
        return suc, reward


    def deb_show(self):
        for i in range(4):
            for j in range(4):
                print(self.board[i, j], end='  ')
            print('')


if __name__ == '__main__':
    c = Core()
    c.deb_show()