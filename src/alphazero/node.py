from typing import Optional

from src import engine
from network import AlphaZero
from src.alphazero.conversions import board_to_tensor, move_to_policy_index


class Node:
    def __init__(
        self,
        state: engine.State,
        parent: Optional["Node"],
        prior_value: float,
        c_puct: float,
        player: int,
    ):
        self.state = state
        self.parent = parent
        self.prior_value = prior_value
        self.c_puct = c_puct
        self.player = player

        self.children: [Node] = []
        self.value = 0
        self.visit_count = 0

    def expand_children(self, network: AlphaZero) -> Optional[float]:
        assert not self.children  # Can only expand once
        board = board_to_tensor(self.state.board)
        board = board.unsqueeze(0)
        policy, value = network(board)
        next_states = engine.get_next_states(self.state.board)

        if len(next_states) == 0:
            return None

        self.children = self._mask_policy_children(policy, next_states)
        return value

    @staticmethod
    def _ucb(parent: "Node", child: "Node") -> float:
        # TODO: Check if UCB formula is correct
        exploration = (
            parent.c_puct
            * child.prior_value
            * (parent.visit_count**0.5 / (1 + child.visit_count))
        )

        if child.visit_count == 0:
            exploitation = 0
        else:
            exploitation = child.value / (1 + child.visit_count)

        return exploration + exploitation

    def best_child(self) -> "Node":
        return max(self.children, key=lambda c: self._ucb(parent=self, child=c))

    def _mask_policy_children(self, policy, children: [engine.State]) -> ["Node"]:
        # TODO: Need to softmax the policy before?
        policy_indexes = [move_to_policy_index(c.move) for c in children]
        masked_policy = policy[:, policy_indexes]
        masked_policy /= masked_policy.sum()

        return [
            Node(
                state,
                parent=self,
                prior_value=masked_policy[:, i],
                c_puct=self.c_puct,
                player=self.player * -1,
            )
            for i, state in enumerate(children)
        ]

    def backpropagate(self, value: float):
        self.visit_count += 1
        self.value += value

        if self.parent:
            self.parent.backpropagate(value * -1)
