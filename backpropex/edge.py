"""
An edge in the network, with weights
"""

from backpropex.node import Node


class Edge:
    """
    `Edge` connects two nodes in the network.

    It holds the weight of the connection between the two nodes.

    The weight is used to calculate a gradient for a weight with respect to the cost function.
    """
    # The weight of this edge.
    weight: float
    previous: Node
    next: Node
    def __init__(self, previous, next, /, *,
                 initial_weight: float=0.0):
        self.previous = previous
        self.next = next
        self.weight = initial_weight

    @property
    def label(self):
        return f'{self.weight:.2f}'

    def __repr__(self):
        return f'Edge({self.previous} -> {self.next})'
