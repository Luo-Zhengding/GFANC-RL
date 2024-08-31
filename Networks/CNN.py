"""
Some neural networks
"""

import numpy as np

import torch
import torch.nn as nn

from torch.distributions.normal import Normal


class ResBlock(nn.Module):
    """
    The ResNet block.
    """

    def __init__(self, prev_channel, channel, conv_kernel, conv_stride, conv_pad):
        super(ResBlock, self).__init__()
        self.res = nn.Sequential(
            nn.Conv1d(in_channels=prev_channel, out_channels=channel, kernel_size=conv_kernel, stride=conv_stride,
                      padding=conv_pad),
            nn.BatchNorm1d(channel),
            nn.ReLU(),
            nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=conv_kernel, stride=conv_stride,
                      padding=conv_pad),
            nn.BatchNorm1d(channel),
        )
        self.bn = nn.BatchNorm1d(channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.res(x)
        if x.shape[1] == identity.shape[1]:
            x += identity
        # repeat the smaller block till it reaches the size of the bigger block
        elif x.shape[1] > identity.shape[1]:
            if x.shape[1] % identity.shape[1] == 0:
                x += identity.repeat(1, x.shape[1] // identity.shape[1], 1)
            else:
                raise RuntimeError("Dims in ResBlock needs to be divisible on the previous dims!!")
        else:
            if identity.shape[1] % x.shape[1] == 0:
                identity += x.repeat(1, identity.shape[1] // x.shape[1], 1)
            else:
                raise RuntimeError("Dims in ResBlock needs to be divisible on the previous dims!!")
            x = identity
        x = self.bn(x)
        x = self.relu(x)
        return x


class CNNRes(nn.Module):
    """
    The 1D CNN.
    """

    def __init__(self, channels=[[128], [128] * 2], conv_kernels=[80, 3], conv_strides=[4, 1], conv_padding=[38, 1],
                 pool_padding=[0, 0], num_classes=15):

        assert len(conv_kernels) == len(channels) == len(conv_strides) == len(conv_padding), "Different kernel channels"
        super(CNNRes, self).__init__()

        # create conv block
        prev_channel = 1
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=prev_channel, out_channels=channels[0][0], kernel_size=conv_kernels[0],
                      stride=conv_strides[0], padding=conv_padding[0]),
            # add batch norm layer
            nn.BatchNorm1d(channels[0][0]),
            # adding ReLU
            nn.ReLU(),
            # adding max pool
            nn.MaxPool1d(kernel_size=4, stride=4, padding=pool_padding[0]),
        )

        # create res
        prev_channel = channels[0][0]
        self.res_blocks = nn.ModuleList()
        for i in range(1, len(channels)):
            # add stacked res layer
            block = []
            for j, conv_channel in enumerate(channels[i]):
                block.append(ResBlock(prev_channel, conv_channel, conv_kernels[i], conv_strides[i], conv_padding[i]))
                prev_channel = conv_channel
            self.res_blocks.append(nn.Sequential(*block))

        # create pool blocks
        self.pool_blocks = nn.ModuleList()
        for i in range(1, len(pool_padding)):
            # adding Max Pool (drops dims by a factor of 4)
            self.pool_blocks.append(nn.MaxPool1d(kernel_size=4, stride=4, padding=pool_padding[i]))

        # global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Sequential(
            nn.Linear(prev_channel, num_classes),
            nn.Sigmoid())  # ! add Sigmoid

    def forward(self, inwav):
        inwav = self.conv_block(inwav)
        for i in range(len(self.res_blocks)):
            # apply conv layer
            inwav = self.res_blocks[i](inwav)
            # apply max_pool
            if i < len(self.pool_blocks): inwav = self.pool_blocks[i](inwav)
        # apply global pooling
        out = self.global_pool(inwav).squeeze()
        out = self.linear(out)
        return out


class SACActor(CNNRes):
    def __init__(self, env, channels=[[128], [128] * 2], conv_kernels=[80, 3], conv_strides=[4, 1],
                 conv_padding=[38, 1], pool_padding=[0, 0], num_classes=15):
        super(SACActor, self).__init__(channels=channels, conv_kernels=conv_kernels, conv_strides=conv_strides,
                                       conv_padding=conv_padding,
                                       pool_padding=pool_padding, num_classes=num_classes)

        # + the same as the self.linear, without the Sigmoid() function.
        self.fc_logstd = nn.Linear(channels[0][0], np.prod(env.action_space.shape))

        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor(np.ones(15) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor(np.ones(15) / 2.0, dtype=torch.float32)
        )

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5

    def forward(self, inwav):
        inwav = self.conv_block(inwav)
        for i in range(len(self.res_blocks)):
            # apply conv layer
            inwav = self.res_blocks[i](inwav)
            # apply max_pool
            if i < len(self.pool_blocks): inwav = self.pool_blocks[i](inwav)
        # apply global pooling
        out = self.global_pool(inwav).squeeze()
        mean = self.linear(out)
        log_std = self.fc_logstd(out)
        log_std = torch.tanh(log_std)
        # * From SpinUp / Denis Yarats
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)

        return mean, log_std

    def get_action(self, inwav):
        mean, log_std = self(inwav)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        if log_prob.ndim == 1:
            log_prob = log_prob.unsqueeze(0)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class SACQNet(CNNRes):
    def __init__(self, env, channels=[[128], [128] * 2], conv_kernels=[80, 3], conv_strides=[4, 1],
                 conv_padding=[38, 1], pool_padding=[0, 0], num_classes=15):
        super(SACQNet, self).__init__(channels=channels, conv_kernels=conv_kernels, conv_strides=conv_strides,
                                      conv_padding=conv_padding,
                                      pool_padding=pool_padding, num_classes=num_classes)

        self.linear = nn.Linear(channels[0][0], 1)

    def forward(self, inwav, action):
        action = action.unsqueeze(1)
        inwav = torch.cat([inwav, action], 2)
        inwav = self.conv_block(inwav)
        for i in range(len(self.res_blocks)):
            # apply conv layer
            inwav = self.res_blocks[i](inwav)
            # apply max_pool
            if i < len(self.pool_blocks): inwav = self.pool_blocks[i](inwav)
        # apply global pooling
        out = self.global_pool(inwav).squeeze()
        out = self.linear(out)
        return out
