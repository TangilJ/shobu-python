from network import AlphaZero
from mcts import AlphaZeroMCTS
import engine


POLICY_OUTPUT_SIZE = (
    16  # passive_source_index
    * 16  # aggressive_source_index
    * 2  # passive_side
    * 2  # aggressive_side
    * 8  # direction
    * 2  # times_moved
)  # = 16384

model = AlphaZero(
    hidden_size=5,
    policy_hidden_size=5,
    policy_output_size=POLICY_OUTPUT_SIZE,
    num_residual_blocks=3,
)

mcts = AlphaZeroMCTS(
    network=model,
    c_puct=2,
    num_sims=100,
)

state = engine.start_state
engine.print_board(state.board)
print("-" * 20)

for i in range(10):
    state = mcts.search(state)

    to_print = state.board
    if i % 2 == 1:
        to_print = engine.reverse_board(to_print)
    engine.print_board(to_print)

    print("-" * 20)
