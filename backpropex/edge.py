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
    from_: Node
    to_: Node
    def __init__(self, from_: Node, to_: Node, /, *,
                 initial_weight: float=0.0):
        self.from_ = from_
        self.to_ = to_
        self.weight = initial_weight


    @property
    def weight(self) -> float:
        return self.to_.layer.weights[self.from_.idx, self.to_.idx]

    @weight.setter
    def weight(self, value: float):
        self.to_.layer.weights[self.from_.idx, self.to_.idx] = value

    @property
    def label(self):
        return f'{self.weight:.2f}'

    def __repr__(self):
        return f'Edge({self.from_} -> {self.to_})={self.weight:.2f}'

__all__ = ['Edge']
