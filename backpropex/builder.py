"""
A builder constructs a neural network from a specification.
"""

from typing import Any, Optional, Sequence
import re

import numpy as np

from backpropex.activation import ACT_ReLU, ACT_Sigmoid
from backpropex.edge import Edge
from backpropex.layer import Layer
from backpropex.protocols import ActivationFunction, Builder, NetProtocol
from backpropex.types import LayerType


        # We are all set up, so let's define our input and output.
def sanitize(name: str) -> str:
    return '_'.join((s for s in re.split(r'[^a-zA-Z0-9_]+', name) if s != ''))

class DefaultBuilder(Builder):
    """
    A builder that constructs a neural network from a specification consisting of
    a list of layer sizes.
    """

    def __call__(self, net: NetProtocol, *layers: int,
                 activations: Optional[Sequence[ActivationFunction]]=None,
                 input_names: Optional[Sequence[str]]=None,
                 output_names: Optional[Sequence[str]]=None,
                 **kwargs: Any) -> None:
        """
        Construct a neural network from a specification.

        :param net: The network to construct.
        :param args: The arguments for the specification.
        :param kwargs: The keyword arguments for the specification.
        """

        if activations is None:
            # Default to ReLU for hidden layers and sigmoid for output
            activations = [ACT_ReLU] * (len(layers) - 1) + [ACT_Sigmoid]

        max_layer_size = max(layers)


        def layer_type(idx: int, nlayers: int = -1):
            match idx:
                case 0:
                    return LayerType.Input
                case _ if idx == nlayers - 1:
                    return LayerType.Output
                case _:
                    return LayerType.Hidden

        def connect_layers(prev: Layer, next: Layer):
            """
            Connect two layers in the network.
            """
            psize = len(prev.nodes)
            nsize = len(next.nodes)
            # Initialize the edge weights by He-et-al initialization
            w = np.random.randn(nsize, psize) * np.sqrt(2 / psize)
            for (pidx, node) in enumerate(prev.nodes):
                for (nidx, next_node) in enumerate(next.real_nodes):
                    edge = Edge(node, next_node, initial_weight=w[nidx][pidx])
                    net.graph.add_edge(node, next_node, edge=edge) # type: ignore

        def node_names(ltype: LayerType):
            if ltype == LayerType.Input:
                if input_names is None:
                    return [f'In[{idx}]' for idx in range(layers[0])]
                return input_names
            if ltype == LayerType.Output:
                if output_names is None:
                    return [f'Out[{idx}]' for idx in range(layers[-1])]
                return output_names
            return None

        def make_layer(idx: int, nodes: int, activation: ActivationFunction):
            ltype = layer_type(idx, len(layers))
            return Layer(nodes,
                    position=idx,
                    activation=activation,
                    max_layer_size=max_layer_size,
                    names=node_names(ltype),
                    layer_type=ltype)

        net.layers = [
            make_layer(idx, nodes, activation)
            for nodes, activation, idx
            in zip(layers, activations, range(len(layers)))
        ]
        # Label the layers
        net.layers[0].label = 'Input'
        net.layers[-1].label = 'Output'
        for idx, layer in enumerate(net.layers[1:-1]):
            layer.label = f'Hidden[{idx}]'
        # Connect the layer nodes
        prev: Layer = net.layers[0]
        for layer in net.layers[1:]:
            connect_layers(prev, layer)
            prev = layer
