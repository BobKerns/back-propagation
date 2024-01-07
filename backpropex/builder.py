"""
A builder constructs a neural network from a specification.
"""

from typing import Any, Optional, Sequence
import re

from backpropex.activation import ACT_ReLU, ACT_Sigmoid
from backpropex.edge import Edge
from backpropex.layer import Layer
from backpropex.protocols import (
    ActivationFunction, Builder, BuilderContext,\
    )
from backpropex.types import LayerType

# We are all set up, so let's define our input and output.
def sanitize(name: str) -> str:
    return '_'.join((s for s in re.split(r'[^a-zA-Z0-9_]+', name) if s != ''))

class DefaultBuilder(Builder):
    """
    A builder that constructs a neural network from a specification consisting of
    a list of layer sizes.
    """
    net: BuilderContext

    def __call__(self, net: BuilderContext, *layers: int,
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
        self.net = net

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

        def connect_layers(from_: Layer, to_: Layer):
            """
            Connect two layers in the network.
            """
            for from_node in from_.nodes:
                for to_node in to_.real_nodes:
                    edge = Edge(from_node, to_node)
                    net.add_edge(edge)

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

        net.add_layers(
            make_layer(idx, nodes, activation)
            for nodes, activation, idx
            in zip(layers, activations, range(len(layers)))
        )
        # Label the layers
        net.layers[0].label = 'Input'
        net.layers[-1].label = 'Output'
        for idx, to_ in enumerate(net.layers[1:-1]):
            to_.label = f'Hidden[{idx}]'
        # Connect the layer nodes
        for from_, to_ in zip(net.layers[0:-1], net.layers[1:]):
            connect_layers(from_, to_)
