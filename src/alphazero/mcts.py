from node import Node
from src import engine
from src.player import Player


class AlphaZeroMCTS(Player):
    def __init__(self, network, c_puct, num_sims):
        self.network = network
        self.c_puct = c_puct
        self.num_sims = num_sims

    def search(self, state: engine.State) -> Node:
        root = Node(state, parent=None, prior_value=0, c_puct=self.c_puct, player=1)

        for _ in range(self.num_sims):
            node = root

            # Select unexpanded node
            while len(node.children) > 0:
                node = node.best_child()

            # Find value of board state
            if node.state.win == engine.Win.OwnWin:
                value = 1
            elif node.state.win == engine.Win.OpponentWin:
                value = -1
            else:
                value = node.expand_children(self.network)
                # Lose if no moves left
                if value is None:
                    value = -1

            # Backpropagate
            node.backpropagate(value)

        return root

    def move(self, state: engine.State) -> engine.State:
        root = self.search(state)
        next_node = max(root.children, key=lambda c: c.visit_count)
        return next_node.state
