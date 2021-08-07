import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(n_states, 10)
        self.fc2 = nn.Linear(10, n_actions)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        out = self.fc2(x)
        return out



class DQN:
    def __init__(self, n_states, n_actions, epsilon=0.9, gamma=0.9):
        self.eval_net = Net(n_states, n_actions)
        self.target_net = Net(n_states, n_actions)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.01)
        self.n_states = n_states
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.gamma = gamma


    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < self.epsilon:
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1].data.numpy()[0]
        else:
            action = np.random.randint(0, self.n_actions)
        return action


    def learn(self, state, action, reward, done, next_state):
        # update target net
        self.target_net.load_state_dict((self.eval_net.state_dict()))

        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        q_eval = self.eval_net(state)[0][action]

        next_state = torch.unsqueeze(torch.FloatTensor(next_state), 0)
        q_next = self.target_net(next_state).detach()

        if done:
            q_target = torch.FloatTensor([reward]).unsqueeze(1)
        else:
            q_target = reward + self.gamma * q_next.max(1)[0].unsqueeze(1)

        loss = self.loss_function(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def show_table(self):
        for i in range(6):
            q = self.eval_net(torch.unsqueeze(torch.FloatTensor([i]), 0)).detach().numpy()
            print(q)