import os
import random
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
from network import AlphaZero
import engine
from src.mcts import AlphaZeroMCTS
from src.conversions import move_to_policy_index, board_to_tensor
from src.node import Node


@dataclass
class EnvConfig:
    policy_size: int
    learning_rate: float
    c_puct: float
    batch_size: int
    iterations: int
    playouts: int
    epochs: int
    simulations: int


@dataclass
class TrainingExample:
    board: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor


@dataclass
class SearchState:
    board: engine.Board
    policy: torch.Tensor
    player: int


class Environment:
    def __init__(self, network: AlphaZero, config: EnvConfig):
        self._network = network
        self._config = config
        self._mcts = AlphaZeroMCTS(network, config.c_puct, config.simulations)

    def _playout(self) -> [TrainingExample]:
        states: [SearchState] = []
        state = engine.start_state
        player = 1

        while True:
            root = self._mcts.search(state)

            policy_idxs = [move_to_policy_index(c.move) for c in root.children]
            visits = [c.visit_count for c in root.children]
            move_probabilities = visits / np.sum(visits)

            policy_vector = torch.zeros(self._config.policy_size)
            policy_vector[policy_idxs] = move_probabilities
            states.append(
                SearchState(board=state.board, policy=policy_vector, player=player)
            )

            next_node = np.random.choice(root.children, p=move_probabilities)
            state = next_node.state
            player *= -1

            if state.win != engine.Win.GameOngoing:
                break

        win_multiplier = 1 if state.win == engine.Win.OwnWin else -1
        return [
            TrainingExample(
                board=board_to_tensor(s.board),
                policy=s.policy,
                value=torch.tensor(s.player * win_multiplier),
            )
            for s in states
        ]

    def _train(self, data: [TrainingExample]):
        optimiser = torch.optim.Adam(
            self._network.parameters(), lr=self._config.learning_rate
        )

        random.shuffle(data)
        for batch in range(0, len(data), self._config.batch_size):
            batch_data = data[batch : batch + self._config.batch_size]

            boards = torch.stack([d.board for d in batch_data])
            policies = torch.stack([d.policy for d in batch_data])
            values = torch.stack([d.value for d in batch_data])

            policy, value = self._network(boards)
            policy_loss = torch.nn.functional.cross_entropy(policy, policies)
            value_loss = torch.nn.functional.mse_loss(value, values)
            loss = policy_loss + value_loss

            optimiser.zero_grad()
            loss.backward()
            self._network.optimiser.step()

    def _save(self, time: str):
        if "models" not in os.listdir():
            os.mkdir("models")
        if time not in os.listdir("models"):
            os.mkdir(f"models/{time}")
        checkpoint_num = len(os.listdir(f"models/{time}"))
        torch.save(self._network, f"models/{time}/checkpoint_{checkpoint_num}.pt")

    def learn(self):
        start_time = datetime.now().strftime("%H%M%S")

        for _ in range(self._config.iterations):
            data = []
            for _ in range(self._config.playouts):
                data += self._playout()

            for _ in range(self._config.epochs):
                self._train(data)

            self._save(start_time)
