from network import AlphaZero
import torch


class Environment:
    @staticmethod
    def train(network: AlphaZero, config: EnvConfig) -> AlphaZero: ...
