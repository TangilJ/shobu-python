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


class AlphaZero(nn.Module):
    def __init__(
        self,
        hidden_size,
        policy_hidden_size,
        num_residual_blocks,
        kernel_size=3,
        padding=1,
    ):
        super().__init__()

        IN_SIZE = 4  # Number of quarters

        self._initial_conv = Conv(IN_SIZE, hidden_size, kernel_size, padding)
        self._residual = nn.ModuleList(
            [
                Residual(hidden_size, kernel_size, padding)
                for _ in range(num_residual_blocks)
            ]
        )
        self._policy_head = Head(
            hidden_size, policy_hidden_size, POLICY_OUTPUT_SIZE, kernel_size, padding
        )
        self._value_head = Head(hidden_size, 3, 1, kernel_size, padding)

    def forward(self, x):
        x = self._initial_conv(x)
        for block in self._residual:
            x = block(x)
        policy = self._policy_head(x)
        value = self._value_head(x)
        return policy, torch.tanh(value)


class Conv(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv3d(in_size, out_size, kernel_size, padding=padding)
        self.norm = nn.BatchNorm3d(out_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.relu(x)


class Head(nn.Module):
    def __init__(self, in_size, hidden_size, output_size, kernel_size, padding):
        super().__init__()
        self.conv = Conv(in_size, hidden_size, kernel_size, padding)
        self.flatten = nn.Flatten()
        self.fc = nn.LazyLinear(output_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return self.fc(x)


class Residual(nn.Module):
    def __init__(self, hidden_size, kernel_size, padding):
        super().__init__()
        self.conv1 = nn.Conv3d(hidden_size, hidden_size, kernel_size, padding=padding)
        self.norm1 = nn.BatchNorm3d(hidden_size)
        self.conv2 = nn.Conv3d(hidden_size, hidden_size, kernel_size, padding=padding)
        self.norm2 = nn.BatchNorm3d(hidden_size)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x += res
        return F.relu(x)
