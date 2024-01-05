# Neural Network Diagram Generator

In support of my paper on AI and copyright, this generates a series of neural network diagrams, step-by-step.

Usage is illustrated in the [Jupyter notebook](notebooks/simple-nn.md).

`Bias` nodes are shown in a way which partially dissoaciates them from the layers. This is accomplished by shifting them halfway between layers, and using a distinctive appearance.

`Layer` instances are labeled and tagged with the `ActivationFunction` used in the `Layer`.

`Node` and `Edge` instances are color-coded to reflect their values and weights, respectively. (`Edge` instances are displayed as lines
labelled with their weights.) Blue is the most negative, red the most positive, passing through light grey at zero.

`Network` instances are callable. The result is a generator, that on every call to `next()`, produces a new diagram for each step of the forward propagation. The value yielded is the label from the diagram, util the output stage is reached. The result from the output will be packaged as a tuple.

When composed with a `Netgraph`, the `Netgraph` is callable in the same way.

You train a `Network` by composing it with a `Trainer`, and calling it
with training data. If you compose this combination with a `Netgraph`,
the combination is callable in the same way as a `Trainer`, and you get a (large) series of graphical diagrams of the training process.

Injecting a `Filter`, either during construction or during invocation, can
narrow down what diagrams are produced.

You can choose from a wide variety of `LossFunction` instances when creating a `Trainer`. The default is `MeanSquaredError`.

When creating a `Network`, its internal structure is created by a `Builder`. The supplied `Builder` takes a list of layer sizes and optionally a list of `ActivationFunction` instances.

The edge weights are then randomized with a `Randomizer`. The supplied
`Randomizer` uses He-et-al randomization.

## Modularity

Some diagrams:

* [Network](doc/network.md)
* [Builder](doc/builder.md)
* [Network + Netgraph](doc/network-netgraph.md)
* [Trainer + Netgraph](doc/trainer-netgraph.md)
* [Step-Centric View](doc/steps.md)
* [User-level Objects](doc/user.md)
