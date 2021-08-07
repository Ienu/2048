#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Copyright (c). All Rights Reserved.
# -----------------------------------------------------
# File Name:        qlearning.py
# Creator:          Wenyu Li
# Version:          0.1
# Created:          2021/08/03
# Description:      qlearning for 2048 game
# Function List:    class Qlearning
# History:
#   <author>      <version>       <time>          <description>
#   Wenyu Li      0.1             2021/08/03       create
#   Wenyu Li      0.2             2021/08/04       fix bugs
#   Wenyu Li      0.3             2021/08/05       use target q method
# -----------------------------------------------------

import torch
import numpy as np
import random
import time

from core import Core
from main import GUI
from model import ValueNet


class Qlearning:
    def __init__(self):
        self.value_net = ValueNet(2)
        self.target_value_net = ValueNet(2)

        # self.value_net.load_state_dict(torch.load('./value_net.pkl'))
        torch.save(self.value_net.state_dict(), './value_net.pkl')
        self.target_value_net.load_state_dict(torch.load('./value_net.pkl'))

        self.episode = 0
        self.explore = 0.3

        self.buffer = []
        self.buffer_capacity = 20
        self.buffer_index = 0


    def value(self, board):
        board_np = np.array(board, dtype=np.float32)
        board_flat = board_np.flatten()
        board_tensor = torch.from_numpy(board_flat)
        values_tensor = self.value_net(board_tensor.detach())
        return values_tensor


    def target_value(self, board):
        board_np = np.array(board, dtype=np.float32)
        board_flat = board_np.flatten()
        board_tensor = torch.from_numpy(board_flat)
        target_values_tensor = self.target_value_net(board_tensor.detach())
        return target_values_tensor


    def action(self, board):
        epsilon = random.random()
        if self.explore <= 0.9:
            self.explore *= 1.001
        print('explore = %.3f' % self.explore)
        if epsilon > self.explore:
            action_num = random.randint(0, 3)
            return action_num

        board_np = np.array(board, dtype=np.float32)
        board_flat = board_np.flatten()
        board_tensor = torch.from_numpy(board_flat)
        values_tensor = self.value(board_tensor.detach())
        values_np = values_tensor.detach().numpy()
        max_idx = np.argmax(values_np.tolist())
        return max_idx


    def train(self):
        # randomly select samples as one batch
        train_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        sample_size = 10#self.buffer_capacity

        indices = np.random.choice(range(10), sample_size)
        sampler1 = torch.utils.data.SubsetRandomSampler(indices)

        print('sampler1 = ', sampler1)
        exit()



        value_loss = self.value_net.loss_function(v, v_.detach(), reward)
        self.value_net.zero_grad()
        value_loss[idx].backward()
        self.value_net.opt_Adam.step()


    def update(self):
        self.episode += 1
        print('episode = ', self.episode)
        if self.episode % 2 == 0:
            torch.save(self.value_net.state_dict(), './value_net.pkl')
            self.target_value_net.load_state_dict(torch.load('./value_net.pkl'))


ql = Qlearning()

def call_back(gui):
    def loop():
        gui._show()
        # produce action

        # # random policy
        # action_num = random.randint(0, 3)

        board_np = np.array(gui.core.board, dtype=np.float32)
        board_flat = board_np.flatten()
        s = torch.from_numpy(board_flat)

        # Q learning
        # v = ql.value(gui.core.board)
        action_num = ql.action(gui.core.board)

        a = action_num

        # interact with env and get reward
        if action_num == 0:
            suc, reward = gui.core.action_up()
            print('UP')
        elif action_num == 1:
            suc, reward = gui.core.action_down()
            print('DOWN')
        elif action_num == 2:
            suc, reward = gui.core.action_left()
            print('LEFT')
        elif action_num == 3:
            suc, reward = gui.core.action_right()
            print('RIGHT')

        # gui.core.deb_show()
        gui.core.emerge()
        gui._show()
        gui.root.update()

        # gui.core.deb_show()

        # judge if done
        # judge state
        done = False

        if gui.core.suc2048:
            print('Done, success')
            done = True
            reward = 1000
        elif gui.core._test():
            print('Done, failed')
            done = True
            reward = -16

        
        if False == suc:
            reward = -16
            # gui.core.emerge()
            # gui._show()

        print('reward = ', reward)
        r = reward
        # v_ = ql.target_value(gui.core.board)

        board_np = np.array(gui.core.board, dtype=np.float32)
        board_flat = board_np.flatten()
        s_ = torch.from_numpy(board_flat)

        sample = {}
        sample['s'] = s
        sample['a'] = a
        sample['r'] = r
        sample['s_'] = s_

        if len(ql.buffer) < ql.buffer_capacity:
            ql.buffer.append(sample)
        else:
            ql.buffer[ql.buffer_index] = sample
            ql.buffer_index += 1
            ql.train()
            ql.update()
            
        # DEBUG
        print('BUFFER: ', ql.buffer)

        # update network
        # ql.train()
        # print('reward = ', reward)

        # ql.update()

        if done:
            # gui.core.emerge()
            # gui._show()
            
            # time.sleep(5)
            gui._new_game()
            print('NEW GAME')
            time.sleep(1)

        # next
        gui.root.after(100, loop)
    
    loop()
    


if __name__ == '__main__':
    gui = GUI(interact=call_back)