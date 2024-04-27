import os
import random
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import torch
import logging

from network import AlphaZero
from src import engine
from src.mcts import AlphaZeroMCTS
from src.conversions import move_to_policy_index, board_to_tensor

logger = logging.getLogger(__name__)


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

            policy_idxs = [move_to_policy_index(c.state.move) for c in root.children]
            visits = [c.visit_count for c in root.children]
            move_probabilities = torch.tensor(visits, dtype=torch.float32) / sum(visits)

            policy_vector = torch.zeros(self._config.policy_size)
            policy_vector[policy_idxs] = move_probabilities
            states.append(
                SearchState(board=state.board, policy=policy_vector, player=player)
            )

            next_node = np.random.choice(root.children, p=move_probabilities.numpy())
            state = next_node.state
            player *= -1

            if state.win != engine.Win.GameOngoing:
                break

        win_multiplier = 1 if state.win == engine.Win.OwnWin else -1
        return [
            TrainingExample(
                board=board_to_tensor(s.board),
                policy=s.policy,
                value=torch.tensor(s.player * win_multiplier, dtype=torch.float32),
            )
            for s in states
        ]

    def _get_loss(self, data: [TrainingExample]):
        boards = torch.stack([d.board for d in data])
        policies = torch.stack([d.policy for d in data])
        values = torch.stack([d.value for d in data])

        policy, value = self._network(boards)
        policy_loss = torch.nn.functional.cross_entropy(policy, policies)
        value_loss = torch.nn.functional.mse_loss(value, values)
        return policy_loss + value_loss

    def _train(self, data: [TrainingExample]):
        optimiser = torch.optim.Adam(
            self._network.parameters(), lr=self._config.learning_rate
        )

        random.shuffle(data)
        for batch in range(0, len(data), self._config.batch_size):
            batch_data = data[batch : batch + self._config.batch_size]
            loss = self._get_loss(batch_data)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    def _get_training_data(self) -> [TrainingExample]:
        data = []
        for playout in range(self._config.playouts):
            data += self._playout()
            logger.info(
                f"Playout {playout + 1}/{self._config.playouts} with data size {len(data)}"
            )
        return data

    def learn(self):
        start_time = datetime.now().strftime("%H%M%S")

        logger.info(f"Training model at {start_time} with config: {self._config}")

        for iteration in range(self._config.iterations):
            logger.info(f"Iteration {iteration + 1}/{self._config.iterations}")

            data = self._get_training_data()
            for epoch in range(self._config.epochs):
                logger.info(f"Epoch {epoch + 1}/{self._config.epochs}")
                self._train(data)

            self._save(start_time)
