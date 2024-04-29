import logging
import os
import pickle
import random
from dataclasses import dataclass

import torch

from src import engine
from src.alphazero.conversions import board_to_tensor
from src.alphazero.network import AlphaZero

logger = logging.getLogger(__name__)

NUM_ITERATIONS = 100
MAX_DEPTH = 300


@dataclass
class TrainingData:
    board: engine.Board
    value: float


@dataclass
class EncodedData:
    board: torch.Tensor
    value: float


def generate(network: AlphaZero):
    logger.info("Generating training data")

    data = []
    for i in range(NUM_ITERATIONS):
        iteration_data = []

        state = engine.start_state
        for depth in range(MAX_DEPTH):
            next_states = engine.get_next_states(state.board)

            # Break if any of next states are terminal states
            if any([s.win != engine.Win.GameOngoing for s in next_states]):
                break

            boards = [x.board for x in next_states]
            nn_input = torch.stack([board_to_tensor(x) for x in boards])
            _, values = network(nn_input)

            iteration_data += [TrainingData(b, v.item()) for b, v in zip(boards, values)]

            state = random.choice(next_states)
        else:
            # Don't add training data for games that go on too long (loops)
            logger.info(f"Skipping iteration {i}")
            break

        logger.info(f"Finished iteration {i} on depth {depth} with size {len(iteration_data)}")
        data += iteration_data

    logger.info(f"Generated training data of size {len(data)}")
    return data


def bitboard_encoding(bitboard: int) -> torch.Tensor:
    encoding = torch.zeros(16)
    b = 0b1000_0000_0000_0000
    i = 0

    while b:
        encoding[i] = (bitboard & b) != 0
        b >>= 1
        i += 1

    return encoding


QUARTER_ENCODING = 4 * 16 * 2
# 4 pieces * 16 squares * 2 players
# = 128 neurons for a quarter

HALF_BOARD_ENCODING = QUARTER_ENCODING * 4
# 4 pieces * 16 squares * 2 players * 4 quarters
# = 512 neurons for whole board for this half

FULL_BOARD_ENCODING = HALF_BOARD_ENCODING * 2


def quarter_encoding(own: int, enemy: int) -> torch.Tensor:
    encoding = torch.zeros(QUARTER_ENCODING)

    stones_left = engine.popcnt(own)

    idx_own = (stones_left - 1) * 16
    idx_enemy = 4 * 16 + idx_own
    encoding[idx_own : idx_own + 16] = bitboard_encoding(own)
    encoding[idx_enemy : idx_enemy + 16] = bitboard_encoding(enemy)

    return encoding


def half_board_encoding(board: engine.Board, first_half: bool) -> torch.Tensor:
    encoding = torch.zeros(HALF_BOARD_ENCODING)

    quarters = [board.top_left, board.top_right, board.bottom_left, board.bottom_right]
    for i, quarter in enumerate(quarters):
        idx = i * QUARTER_ENCODING
        if first_half:
            own, enemy = quarter.own, quarter.enemy
        else:
            own, enemy = quarter.enemy, quarter.own
        encoding[idx : idx + QUARTER_ENCODING] = quarter_encoding(own, enemy)

    return encoding


def full_board_encoding(board: engine.Board) -> torch.Tensor:
    encoding = torch.zeros(FULL_BOARD_ENCODING)

    encoding[:HALF_BOARD_ENCODING] = half_board_encoding(board, first_half=True)
    encoding[HALF_BOARD_ENCODING:] = half_board_encoding(board, first_half=False)

    return encoding


def encode_data(data: [TrainingData]) -> [EncodedData]:
    logger.info(f"Encoding data of size {len(data)}")
    return [EncodedData(full_board_encoding(d.board), d.value) for d in data]


def save_data(data: [EncodedData], path: str):
    with open(path, "wb") as f:
        pickle.dump(data, f)
        logger.info(f"Saved data to: {path}")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )

    nn = torch.load("models/023243/checkpoint_19.pt")
    training_data = generate(nn)
    encoded = encode_data(training_data)

    if "nnue" not in os.listdir():
        os.mkdir("nnue")
    save_data(encoded, "nnue/training_data.pkl")
