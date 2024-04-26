import torch
import engine
from src import engine


def bitboard_to_tensor(bitboard: int) -> torch.Tensor:
    tensor = torch.empty((4, 4))
    stone = 0b1000_0000_0000_0000
    for i in range(4):
        for j in range(4):
            tensor[i, j] = (bitboard & stone) != 0
            stone >>= 1
    return tensor


def board_to_tensor(board: engine.Board) -> torch.Tensor:
    # 4 quarters with 3 planes each: own, enemy, empty spaces
    tensor = torch.empty((1, 4, 3, 4, 4))
    # TODO: Also experiment with 16x16 instead of splitting into quarters with Conv2d

    quarters = [board.top_left, board.top_right, board.bottom_left, board.bottom_right]
    for i, quarter in enumerate(quarters):
        tensor[0][i][0] = bitboard_to_tensor(quarter.own)
        tensor[0][i][1] = bitboard_to_tensor(quarter.enemy)
        tensor[0][i][2] = bitboard_to_tensor(~(quarter.own | quarter.enemy))

    return tensor


def move_to_policy_index(move: engine.Move) -> int:
    # noinspection PyTypeChecker
    return (
        move.passive_source_index  # 16
        + move.aggressive_source_index * 16  # 16
        + int(move.passive_side) * 16 * 16  # 2
        + int(move.aggressive_side) * 16 * 16 * 2  # 2
        + int(move.direction) * 16 * 16 * 2 * 2  # 8
        + move.times_moved * 16 * 16 * 2 * 2 * 8  # 2
    )
