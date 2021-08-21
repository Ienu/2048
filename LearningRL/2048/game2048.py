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
        self.cnn = CNN()

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

        # reward -= self.cnn._eval(self.board)
        # print('up reward = %d' % reward)
        ext_reward = 0
        for i in range(self.size):
            for j in range(self.size):
                ext_reward += self.board[i, j] * self.weights[i, j]

        # ext_reward2 = 0
        # for i in range(self.size):
        #     for j in range(self.size - 1):
        #         ext_reward2 += math.fabs(self.board[i, j] - self.board[i, j + 1])
                # ext_reward2 += math.fabs(self.board[j, i] - self.board[j + 1, i])

        return suc, ext_reward #+ reward / 2


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

        # reward -= self.cnn._eval(self.board)
        # print('down reward = %d' % reward)
        ext_reward = 0
        for i in range(self.size):
            for j in range(self.size):
                ext_reward += self.board[i, j] * self.weights[i, j]

        # ext_reward2 = 0
        # for i in range(self.size):
        #     for j in range(self.size - 1):
        #         ext_reward2 += math.fabs(self.board[i, j] - self.board[i, j + 1])
                # ext_reward2 += math.fabs(self.board[j, i] - self.board[j + 1, i])

        return suc, ext_reward #+ reward / 2


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
        # reward -= self.cnn._eval(self.board)
        ext_reward = 0
        for i in range(self.size):
            for j in range(self.size):
                ext_reward += self.board[i, j] * self.weights[i, j]

        # ext_reward2 = 0
        # for i in range(self.size):
        #     for j in range(self.size - 1):
        #         ext_reward2 += math.fabs(self.board[i, j] - self.board[i, j + 1])
                # ext_reward2 += math.fabs(self.board[j, i] - self.board[j + 1, i])

        return suc, ext_reward #+ reward / 2


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
        # reward -= self.cnn._eval(self.board)
        ext_reward = 0
        for i in range(self.size):
            for j in range(self.size):
                ext_reward += self.board[i, j] * self.weights[i, j]

        # ext_reward2 = 0
        # for i in range(self.size):
        #     for j in range(self.size - 1):
        #         ext_reward2 += math.fabs(self.board[i, j] - self.board[i, j + 1])
                # ext_reward2 += math.fabs(self.board[j, i] - self.board[j + 1, i])

        return suc, ext_reward #+ reward / 2


    def deb_show(self):
        for i in range(self.size):
            for j in range(self.size):
                print(self.board[i, j], end='  ')
            print('')



class Game2048(gym.Env, tk.Tk):
    def __init__(self, c=Core(), interact=None):
        logging.debug('<INIT FUNC>')
        super(Game2048, self).__init__()

        self.seed()

        self.core = c

        self.labels = []
        self.first_win = False

        self.title('2048')
        self.geometry('600x800+10+10')

        self._build_env()


    def step(self, action):
        logging.debug('<STEP FUNC>')
        reward = -10
        suc = False
        if action == 0: # up
            suc, reward = self.core.action_up()
        elif action == 1: # left
            suc, reward = self.core.action_left()
        elif action == 2: # right
            suc, reward = self.core.action_right()
        elif action == 3: # down
            suc, reward = self.core.action_down()

        if suc:
            self.core.emerge()
        
        state = self.core.board
        done = self.core._test()
        if done:
            reward = -100

        return state, reward, done, {}


    def reset(self):
        logging.debug('<RESET FUNC>')
        self.first_win = False
        self.core.reset()
        self._show()
        state = self.core.board
        return state


    def render(self, mode='human', fresh_time=0.1):
        logging.debug('<RENDER FUNC>')
        self._show()
        self.update()
        time.sleep(fresh_time)


    def close(self):
        logging.debug('<CLOSE FUNC>')


    def seed(self, seed=None):
        logging.debug('<SEED FUNC>')
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _get_image(self, file_name, width, height):
        im = Image.open(file_name).resize((width, height))
        return ImageTk.PhotoImage(im)


    def _color(self, r, g, b):
        color_str = '#%02x%02x%02x' % (int(r), int(g), int(b))
        return color_str


    def _block(self, value):
        # background color
        color_list = ['#000000', '#eee4da', '#ede0c8', '#f2b179', '#f59563', '#f67c5f', 
            '#f6623d', '#edcf72', '#eccb61', '#edc850', '#edc53f', '#edc22e']

        if value <= 2048:
            bg_color_str = color_list[int(math.log(value) / math.log(2))]
        else:
            bg_color_str = color_list[0]

        # front color
        fg_color_str = self._color(250, 248, 239)
        if value <= 4:
            fg_color_str = self._color(119, 110, 101)

        # font size
        font_size = 48
        if value > 100 and value < 1000:
            font_size = 36
        elif value > 1000 and value < 10000:
            font_size = 24
        elif value > 10000:
            font_size = 12

        # global im_block
        im_block = self._get_image('block.png', 100, 100)
        label = tk.Label(self.canvas, image=im_block, bg=bg_color_str, width=100, height=100, 
            text=str(value), font='Helvetica -%d bold' % font_size, fg=fg_color_str, compound=tk.CENTER)

        return label
        

    def _build_env(self):
        self.canvas = tk.Canvas(self,
            height=800, width=600
        )
        global im_canvas
        im_canvas = self._get_image('2048bkg.png', 600, 800)
        self.canvas.create_image(300, 400, image=im_canvas)
        self.canvas.place(x=0, y=0)


    def _show(self):
        for lab in self.labels:
            lab.destroy()

        for row in range(self.core.size):
            for col in range(self.core.size):
                value = self.core.board[row, col]
                if value != 0:
                    label = self._block(value)
                    label.place(x=64+col*121, y=237+row*121)
                    self.labels.append(label)




