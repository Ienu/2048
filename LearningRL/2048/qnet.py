import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging


class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(n_states, 10)
        self.fc2 = nn.Linear(10, n_actions)
        self.fc3 = nn.Linear(10, 10)

        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3.weight.data.normal_(0, 0.1)


    def forward(self, x):
        x = torch.log(x + 1) / 16
        x = self.fc1(x + 1)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        out = self.fc2(x)
        return out



class DQN:
    def __init__(self, n_states, n_actions, epsilon=0.9, gamma=0.9):
        self.eval_net = Net(n_states, n_actions)
        self.target_net = Net(n_states, n_actions)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.01)

        self.memory_count = 0
        self.memory_capacity = 256
        self.memory = np.zeros([self.memory_capacity, n_states * 2 + 2])
        self.cost = []
        self.learn_step = 0
        self.learn_update_count = 256

        self.batch_size = 256

        self.n_states = n_states
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.ex_eps = 0.1


    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < min(self.epsilon, self.ex_eps):
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1].data.numpy()[0]
        else:
            action = np.random.randint(0, self.n_actions)

        self.ex_eps *= 1.01
        logging.debug('eps = ', self.ex_eps)
        return action


    def learn(self, state, action, reward, done, next_state):
        # update target net
        if self.learn_step % self.learn_update_count == 0:
            self.target_net.load_state_dict((self.eval_net.state_dict()))
        self.learn_update_count += 1

        # store transition
        self._store_transition(state, action, reward, next_state)

        # memory full
        if self.memory_count >= self.memory_capacity:
            self.memory_count = 0
            # use memory for batch data training
            sample_index = np.random.choice(self.memory_capacity, self.batch_size, replace=False)
            memory = self.memory[sample_index, :]   # state action reward next

            State = torch.FloatTensor(memory[:, :self.n_states])
            Action = torch.LongTensor(memory[:, self.n_states:self.n_states+1])
            Reward = torch.FloatTensor(memory[:, self.n_states+1:self.n_states+2])
            Next_State = torch.FloatTensor(memory[:, -self.n_states:])

            logging.debug('State = ', State)
            logging.debug('Action = ', Action)
            logging.debug('Reward = ', Reward)
            logging.debug('Next State = ', Next_State)

            Q_eval = self.eval_net(State).gather(1, Action)
            Q_next = self.target_net(Next_State).detach()
            logging.debug('Q_eval = ', Q_eval)
            logging.debug('Q_next = ', Q_next)

            if done:
                Q_target = Reward
            else:
                Q_target = Reward + self.gamma * Q_next.max(1)[0].unsqueeze(1)
            logging.debug('Q_target = ', Q_target)

            # self._show_table()

            loss = self.loss_function(Q_eval, Q_target)
            logging.debug('loss = ', loss)
            print('loss = ', loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # self._show_table()


    def _store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, action, reward, next_state))
        index = self.memory_count % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_count += 1


    def show_table(self):
        table = []
        for i in range(16):
            row = i // 4 - 2
            col = i % 4 - 2

            q = self.eval_net(torch.unsqueeze(torch.FloatTensor([row, col]), 0)).detach().numpy()
            table.append(q[0])
            print(q)
        return table