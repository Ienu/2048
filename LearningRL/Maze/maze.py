import numpy as np
import time
import logging

import gym
from gym import spaces
from gym.utils import seeding

import tkinter as tk


class Maze(gym.Env, tk.Tk):
    def __init__(self):
        logging.debug('<INIT FUNC>')
        super(Maze, self).__init__()

        self.n_actions = 4
        self.n_states = 2

        self.min_position = 0
        self.max_position = 0

        self.low = np.array(
            [self.min_position, self.min_position], 
            dtype=np.integer
        )
        self.high = np.array(
            [self.max_position, self.max_position],
            dtype=np.integer
        )

        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
            self.low, self.high, dtype=np.integer
        )

        self.title('Maze')
        self.geometry('160x160')
        self.maze_height = 4
        self.maze_width = 4
        self.unit = 40

        self.seed()

        self._build_env()


    def step(self, action):
        logging.debug('<STEP FUNC>')
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0: # up
            if s[1] > self.unit:
                base_action[1] -= self.unit
        elif action == 1: # down
            if s[1] < (self.maze_height - 1) * self.unit:
                base_action[1] += self.unit
        elif action == 2:  # right
            if s[0] < (self.maze_width - 1) * self.unit:
                base_action[0] += self.unit
        elif action == 3:  # left
            if s[0] > self.unit:
                base_action[0] -= self.unit
        
        self.canvas.move(self.rect, base_action[0], base_action[1])
        next_coords = self.canvas.coords(self.rect)

        if next_coords == self.canvas.coords(self.oval):
            reward = 1
            print('Victory')
            done = True
        elif next_coords in [self.canvas.coords(self.hell)]:
            reward = -1
            print('Defeat')
            done = True
        else:
            reward = 0
            done = False

        s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / (self.maze_height * self.unit)

        return s_, reward, done, {}


    def reset(self):
        logging.debug('<RESET FUNC>')
        self.update()
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red'
        )
        return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / (self.maze_height * self.unit)


    def render(self, mode='humna', fresh_time=0.1):
        logging.debug('<RENDER FUNC>')
        self.update()
        time.sleep(fresh_time)

    
    def close(self):
        logging.debug('<CLOSE FUNC>')


    def seed(self, seed=None):
        logging.debug('<SEED FUNC>')
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _build_env(self):
        self.canvas = tk.Canvas(self, bg='white',
            height=self.maze_height*self.unit,
            width=self.maze_width*self.unit)

        for c in range(0, self.maze_width*self.unit, self.unit):
            x0, y0, x1, y1 = c, 0, c, self.maze_height * self.unit
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self.maze_height * self.unit, self.unit):
            x0, y0, x1, y1 = 0, r, self.maze_width * self.unit, r
            self.canvas.create_line(x0, y0, x1, y1)

        origin = np.array([20, 20])
        hell_center = origin + np.array([self.unit * 2, self.unit])
        self.hell = self.canvas.create_rectangle(
            hell_center[0] - 15, hell_center[1] - 15,
            hell_center[0] + 15, hell_center[1] + 15,
            fill='black'
        )
        oval_center = origin + self.unit * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow'
        )
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red'
        )
        self.canvas.pack()


    def show_prob(self, table):
        for i in range(16):
            if i == 9 or i == 10:
                continue

            origin = np.array([20, 20])
            # print('table = ', table[i])
            row = i // 4
            col = i % 4

            origin[0] += row * self.unit
            origin[1] += col * self.unit

            # print('origin = ', origin)

            min_t = min(table[i])
            max_t = max(table[i])
            
            if max_t == 0:
                value = [128, 128, 128, 128]
            else:
                value = table[i] - min_t
                value *= 255 / (max_t - min_t)
                # value = int(table[i])
                # print('value = ', value)

            self.circle = self.canvas.create_oval(
                origin[0] - 5, origin[1] - 15,
                origin[0] + 5, origin[1] - 5,
                fill='#00%02x00' % int(value[0])
            )
            self.circle = self.canvas.create_oval(
                origin[0] - 5, origin[1] + 5,
                origin[0] + 5, origin[1] + 15,
                fill='#00%02x00' % int(value[1])
            )
            self.circle = self.canvas.create_oval(
                origin[0] + 5, origin[1] - 5,
                origin[0] + 15, origin[1] + 5,
                fill='#00%02x00' % int(value[2])
            )
            self.circle = self.canvas.create_oval(
                origin[0] - 15, origin[1] - 5,
                origin[0] - 5, origin[1] + 5,
                fill='#00%02x00' % int(value[3])
            )
            self.canvas.pack()
