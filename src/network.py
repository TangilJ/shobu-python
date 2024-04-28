from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

POLICY_OUTPUT_SIZE = (
    16  # passive_source_index
    * 16  # aggressive_source_index
    * 2  # passive_side
    * 2  # aggressive_side
    * 8  # direction
    * 2  # times_moved
)  # = 16384


@dataclass
class ModelConfig:
    hidden_size: int
    value_hidden_size: int
    num_residual_blocks: int


class AlphaZero(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        IN_SIZE = 4  # Number of quarters

        self.initial_conv = ConvBlock(IN_SIZE, config.hidden_size, kernel_size=3)
        self.residual = nn.ModuleList(
            [Residual(config.hidden_size) for _ in range(config.num_residual_blocks)]
        )
        self.policy_head = PolicyHead(config.hidden_size)
        self.value_head = ValueHead(config.hidden_size, config.value_hidden_size)

    def forward(self, x):
        x = self.initial_conv(x)
        for block in self.residual:
            x = block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size):
        super().__init__()
        self.conv = nn.Conv3d(in_size, out_size, kernel_size, padding="same")
        self.norm = nn.BatchNorm3d(out_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.relu(x)


class PolicyHead(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.conv_block = ConvBlock(in_size, 2, kernel_size=1)
        self.flatten = nn.Flatten()
        self.fc = nn.LazyLinear(POLICY_OUTPUT_SIZE)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.softmax(x)


class ValueHead(nn.Module):
    def __init__(self, in_size, linear_hidden_size):
        super().__init__()
        self.conv_block = ConvBlock(in_size, 1, kernel_size=1)
        self.flatten = nn.Flatten()
        self.fc = nn.LazyLinear(linear_hidden_size)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(linear_hidden_size, 1)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc_out(x)
        return torch.tanh(x)


class Residual(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.conv_block = ConvBlock(hidden_size, hidden_size, kernel_size=3)
        self.conv2 = nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding="same")
        self.norm2 = nn.BatchNorm3d(hidden_size)

    def forward(self, x):
        res = x
        x = self.conv_block(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x += res
        return F.relu(x)
