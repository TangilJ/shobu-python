from dataclasses import dataclass


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
