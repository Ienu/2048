import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.gamma = 0.99
        self.learning_rate_value = 1e-4

        self.value_layer1 = nn.Linear(16, 256, bias=True)
        self.value_layer2 = nn.Linear(256, 256, bias=True)
        self.value_layer3 = nn.Linear(256, 256, bias=True)
        self.value_layer4 = nn.Linear(256, 4, bias=True)

        self.init_weights()
        self.optimizer()


    def forward(self, x):
        # x = torch.mul(x, self.norm_matrix)

        x = self.value_layer1(x)
        x = F.elu(x)
        x = self.value_layer2(x)
        x = F.elu(x)
        x = self.value_layer3(x)
        x = F.elu(x)
        x = self.value_layer4(x)
        x = F.softplus(x)
        # print('x = ', x)

        return x


    def loss_function(self, v, v_, L):
        TD_target = L + self.gamma * v_
        TD_error = torch.pow(TD_target - v, 2)
        # print('TD error = ', TD_error)
        # loss = torch.mean(TD_error)
        # print('loss = ', loss)
        # ones_tensor = torch.ones_like(TD_error)
        # print('ones_tensor = ', ones_tensor)
        return TD_error


    def optimizer(self):
        self.opt_Adam = torch.optim.Adam(self.parameters(),
            lr=self.learning_rate_value,
            betas=(0.9, 0.99),
            weight_decay=0)


    def init_weights(self):
        if isinstance(self, nn.Linear):
            weight_shape = list(self.weight.data.size())
            fan_in = weight_shape[1]
            fan_out = weight_shape[0]
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            self.weight.data.uniform_(-w_bound, w_bound)
            self.bias.data.fill_(0)
        elif isinstance(self, nn.BatchNorm1d):
            self.weight.data.fill_(1)
            self.bias.data.zero_()