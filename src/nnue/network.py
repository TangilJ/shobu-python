from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    hidden_1: int
    hidden_2: int
    hidden_3: int


# Input size is based on HalfKP format but with piece positions
# fixed to the number of stones left on the board it's on
INPUT_SIZE = (
        4  # pieces
        * 16  # squares
        * 4  # quarterboards
        * 2  # players
        * 2  # halves
)  # = 1024


class NNUE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.initial = nn.Linear(INPUT_SIZE, config.hidden_1)
        self.hidden_1 = nn.Linear(config.hidden_1, config.hidden_2)
        self.hidden_2 = nn.Linear(config.hidden_2, config.hidden_3)
        self.output = nn.Linear(config.hidden_3, 1)
        self.relu = lambda x: torch.clamp(x, min=0, max=1)  # Clipped ReLU

    def forward(self, x):
        x = self.relu(self.initial(x))
        x = self.relu(self.hidden_1(x))
        x = self.relu(self.hidden_2(x))
        return self.output(x)
