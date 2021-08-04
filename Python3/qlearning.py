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


    def value(self, board):
        board_np = np.array(board, dtype=np.float32)
        # print('board_np = ', board_np)
        board_flat = board_np.flatten()
        # print('board_flat = ', board_flat)
        board_tensor = torch.from_numpy(board_flat)
        values_tensor = self.value_net(board_tensor.detach())
        return values_tensor


    def action(self, board):
        epsilon = random.random()
        if epsilon > 0.9:
            action_num = random.randint(0, 3)
            return action_num

        # print('board = ', board)
        board_np = np.array(board, dtype=np.float32)
        # print('board_np = ', board_np)
        board_flat = board_np.flatten()
        # print('board_flat = ', board_flat)
        board_tensor = torch.from_numpy(board_flat)
        values_tensor = self.value(board_tensor.detach())
        # print('values_tensor = ', values_tensor)
        values_np = values_tensor.detach().numpy()
        # print('values_np = ', values_np)
        max_idx = np.argmax(values_np.tolist())
        # print('max_idx = ', max_idx)
        return max_idx


    def train(self, v, v_, reward, idx):
        value_loss = self.value_net.loss_function(v, v_.detach(), reward)

        # ones_tensor = np.zeros([4])
        
        # ones_tensor[idx] = 1
        # ones_tensor = torch.from_numpy(ones_tensor)
        # print('ones_tensor = ', ones_tensor)
        # self.value_loss_history[self.step] = value_loss.detach().numpy()

        self.value_net.zero_grad()
        # value_loss.backward(ones_tensor, retain_graph=False)
        value_loss[idx].backward()
        # torch.nn.utils.clip_grad_norm_(value_net.parameters(), self.clip_norm)
        self.value_net.opt_Adam.step()


ql = Qlearning()

def call_back(gui):
    def loop():
        gui._show()
        # produce action

        # # random policy
        # action_num = random.randint(0, 3)

        # Q learning
        v = ql.value(gui.core.board)
        action_num = ql.action(gui.core.board)

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

        gui.core.deb_show()
        # time.sleep(5)
        # print('show')

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

        v_ = ql.value(gui.core.board)
        # update network
        ql.train(v, v_, reward + 16, action_num)
        # print('reward = ', reward)

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