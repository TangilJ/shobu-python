from network import AlphaZero
from dataclasses import dataclass
import torch


@dataclass
class AdamConfig:
    learning_rate: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float


@dataclass
class SGDConfig:
    learning_rate: float
    momentum: float
    dampening: float
    weight_decay: float


@dataclass
class EnvConfig:
    optimiser: AdamConfig | SGDConfig
    epochs: float
    simulations: int


class Environment:
    @staticmethod
    def train(network: AlphaZero, config: EnvConfig) -> AlphaZero: ...
