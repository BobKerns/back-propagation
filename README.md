# Neural Network Diagram Generator

In support of my paper on AI and copyright, this generates a series of neural network diagrams, step-by-step.

Usage is illustrated in the [Jupyter notebook](notebooks/simple-nn.md).

Bias nodes are shown in a way which partially dissoaciates them from the layers. This is accomplished by shifting them halfway between layers, and using a distinctive appearance.

Layers are labeled and tagged with the activation function used in the layer

Nodes and edges are color-coded to reflect their values and weights, respectively. Blue is the most negative, red the most positive, passing through light grey at zero.

Networks are callable. The result is a generator, that on every call to `next()`, produces a new diagram for each step of the forward propagation. The value yielded is the label from the diagram, util the output stage is reached. The result from the output will be packaged as a tuple.

## Modularity

### Network

```mermaid
classDiagram
  Node <|-- Input
  Hidden --|> Node
  Output --|> Node
  Bias --|> Node

  Edge --> Input : previous
  Edge --> Hidden : previous
  Edge --> Bias : prevous
  Edge --> Hidden : next
  Edge --> Output : next

  Layer "1" --* Bias : [Except output]
  Layer "1..*" --* Input : real[Input]
  Layer "0..*" --* Hidden : n real[Hidden]
  Layer "0..*" --* Output : real[Output]
  Layer "0..*" --* Node : nodes

  namespace concrete_nodes {
    class Bias {
        + float value = 1.0
        + int position = 0
    }
    class Output {
        + str label
        - float[] gradient
    }
    class Hidden {
        - float[] gradient
    }
    class Input {
        + str label
    }
  }
  class Node {
      + int position
      + float value
  }
  class Edge {
    float weight
  }
  class Layer {
    + int position
    + str label
  }
  class ActivationFunction {
    eval(float) float
    derivative(float) float
  }
  class Graph {
    nodes() Node[]
    edges() Edge[]
    edges_in(Node) Edge[]
    edges_out(Node) Edge[]
  }
  class Network {
    + str name
    eval(float[]) float[]
    nodes() Node[]
    real_nodes() Node[]
    edges() Edge[]
    edges_in(Node) Edge[]
    edges_out(Node) Edge[]
  }
  Network --> InitStep : yield
  Network --> ForwardStep : yield
  Network --> OutputStep : yield
  class Step {
    + StepType type
  }
  class InitStep {
    + float[] weights
  }
  class EvalStep {
    + StepType type
    + float[] values
  }
  class ForwardStep {

  }
  class OutputStep {

  }
  InitStep --|> Step
  EvalStep --|> Step
  ForwardStep --|> EvalStep
  OutputStep --|> EvalStep
  EvalStep --> Layer
  Node "1" --o ActivationFunction
  Layer "1" --o ActivationFunction
  Network "2..*" --* Layer : Layers
  Network "0..*" --* Layer : Hidden
  Network "1" --* Layer : Input
  Network "1" --* Layer : Output
  Network "1" --* Graph
  Graph "..*" --* Node
  Graph "..*" --* Edge

  class InputData {
    float[] data
  }

  class OutputData {
    float[] data
  }
  InputData ..> Network : Invoke
  Network ..> OutputData : Produce
```

### Builder

```mermaid
classDiagram
  Node <|-- Input
  Hidden --|> Node
  Output --|> Node
  Bias --|> Node

  Edge --> Input : previous
  Edge --> Hidden : previous
  Edge --> Bias : prevous
  Edge --> Hidden : next
  Edge --> Output : next

  Layer "1" --* Bias : [Except output]
  Layer "1..*" --* Input : real[Input]
  Layer "0..*" --* Hidden : n real[Hidden]
  Layer "0..*" --* Output : real[Output]
  Layer "0..*" --* Node : nodes

  namespace concrete_nodes {
    class Bias {
        + float value = 1.0
        + int position = 0
    }
    class Output {
        + str label
        - float[] gradient
    }
    class Hidden {
        - float[] gradient
    }
    class Input {
        + str label
    }
  }
  class Node {
      + int position
      + float value
  }
  class Edge {
    float weight
  }
  class Layer {
    + int position
    + str label
  }
  class ActivationFunction {
    eval(float) float
    derivative(float) float
  }
  class Network {
    + str name
    eval(float[]) float[]
    nodes() Node[]
    real_nodes() Node[]
    edges() Edge[]
    edges_in(Node) Edge[]
    edges_out(Node) Edge[]
  }
  Node "1" --o ActivationFunction
  Layer "1" --o ActivationFunction
  Network "2..*" --* Layer : Layers
  Network "0..*" --* Layer : Hidden
  Network "1" --* Layer : Input
  Network "1" --* Layer : Output

  Builder <--> Network
  Builder --> Input
  Builder --> Hidden
  Builder --> Output
  Builder --* Randomizer
  Randomizer --> Edge
  class Builder {
    + specification
  }
  class Randomizer {

  }
```

### Netgraph

```mermaid
classDiagram
  Node <|-- Input
  Hidden --|> Node
  Output --|> Node
  Bias --|> Node

  Edge --> Input : previous
  Edge --> Hidden : previous
  Edge --> Bias : prevous
  Edge --> Hidden : next
  Edge --> Output : next

  Layer "1" --* Bias : [Except output]
  Layer "1..*" --* Input : real[Input]
  Layer "0..*" --* Hidden : n real[Hidden]
  Layer "0..*" --* Output : real[Output]
  Layer "0..*" --* Node : nodes

  namespace concrete_nodes {
    class Bias {
        + float value = 1.0
        + int position = 0
    }
    class Output {
        + str label
        - float[] gradient
    }
    class Hidden {
        - float[] gradient
    }
    class Input {
        + str label
    }
  }
  class Node {
      + int position
      + float value
  }
  class Edge {
    float weight
  }
  class Layer {
    + int position
    + str label
  }
  class ActivationFunction {
    eval(float) float
    derivative(float) float
  }
  class Network {
    + str name
    eval(float[]) float[]
    nodes() Node[]
    real_nodes() Node[]
    edges() Edge[]
    edges_in(Node) Edge[]
    edges_out(Node) Edge[]
  }
  Network --> InitStep : yield
  Network --> ForwardStep : yield
  Network --> OutputStep : yield
  class Step {
    + StepType type
  }
  class InitStep {
    + float[] weights
  }
  class EvalStep {
    + StepType type
    + float[] values
  }
  class ForwardStep {

  }
  class OutputStep {

  }
  InitStep --|> Step
  EvalStep --|> Step
  ForwardStep --|> EvalStep
  OutputStep --|> EvalStep
  EvalStep --> Layer
  Node "1" --o ActivationFunction
  Layer "1" --o ActivationFunction
  Network "2..*" --* Layer : Layers
  Network "0..*" --* Layer : Hidden
  Network "1" --* Layer : Input
  Network "1" --* Layer : Output

  class Netgraph {
    eval(float[]) float[]
  }
  InitStep ..> Netgraph : receive
  ForwardStep ..> Netgraph : receive
  OutputStep ..> Netgraph : receive
  Netgraph ..> InitStep : yield
  Netgraph ..> ForwardStep : yield
  Netgraph ..> OutputStep : yield
  class Filter {
    prefilter(StepType) bool
    filter(StepType) bool
    postfilter(StepType) bool
  }
  Netgraph --* Filter
  Network --* Filter

  class Image {

  }
  Netgraph "*" --> Image : Produce

  class InputData {
    float[] data
  }

  class OutputData {
    float[] data
  }
  InputData ..> Netgraph : Invoke
  InputData ..> Network : Invoke
  Netgraph ..> OutputData : Produce
  Network ..> OutputData : Produce
```

### Trainer + Netgraph

```mermaid
classDiagram
  Node <|-- Input
  Hidden --|> Node
  Output --|> Node
  Bias --|> Node

  Edge --> Input : previous
  Edge --> Hidden : previous
  Edge --> Bias : prevous
  Edge --> Hidden : next
  Edge --> Output : next

  Layer "1" --* Bias : [Except output]
  Layer "1..*" --* Input : real[Input]
  Layer "0..*" --* Hidden : n real[Hidden]
  Layer "0..*" --* Output : real[Output]
  Layer "0..*" --* Node : nodes

  namespace concrete_nodes {
    class Bias {
        + float value = 1.0
        + int position = 0
    }
    class Output {
        + str label
        - float[] gradient
    }
    class Hidden {
        - float[] gradient
    }
    class Input {
        + str label
    }
  }
  class Node {
      + int position
      + float value
  }
  class Edge {
    float weight
  }
  class Layer {
    + int position
    + str label
  }
  class ActivationFunction {
    eval(float) float
    derivative(float) float
  }
  class Network {
    + str name
    eval(float[]) float[]
    nodes() Node[]
    real_nodes() Node[]
    edges() Edge[]
    edges_in(Node) Edge[]
    edges_out(Node) Edge[]
  }
  Network --> InitStep : yield
  Network --> ForwardStep : yield
  Network --> OutputStep : yield
  class Step {
    + StepType type
  }
  class InitStep {
    + float[] weights
  }
  class EvalStep {
    + StepType type
    + float[] values
  }
  class ForwardStep {

  }
  class OutputStep {

  }
  class TrainStep {
    + float[] input
    + float[] expected
  }
  TrainStep --|> Step
  class ForwardTrainStep {

  }
  ForwardTrainStep --|> EvalStep
  ForwardTrainStep --|> TrainStep
  class OutputTrainStep {

  }
  OutputTrainStep --|> OutputStep
  OutputTrainStep --|> TrainStep
  class LossTrainStep {
    + float loss
  }
  LossTrainStep --|> ForwardTrainStep
  class BackwardTrainStep {
    + float loss
    + float[] gradient
    + float[] adjustment
  }
  BackwardTrainStep --|> TrainStep
  InitStep --|> Step
  EvalStep --|> Step
  FowardStep --|> EvalStep
  OutputStep --|> EvalStep
  EvalStep --> Layer
  TrainStep --> Layer
  Node "1" --o ActivationFunction
  Layer "1" --o ActivationFunction
  Network "2..*" --* Layer : Layers
  Network "0..*" --* Layer : Hidden
  Network "1" --* Layer : Input
  Network "1" --* Layer : Output

  class Netgraph {
    eval(float[]) float[]
    train1(float[] input, float[] expected)
    train(TrainingData)
  }
  InitStep ..> Trainer : receive
  ForwardStep ..> Trainer : receive
  OutputStep ..> Trainer : reeive
  Trainer ..> InitStep : yield
  Trainer ..> ForwardTrainStep : yield
  Trainer ..> OutputTraiStep : yield
  Trainer ..> LossTrainStep : yield
  Trainer ..> BackwardTrainStep : yield
  InitStep ..> Netgraph : recieve
  ForwardTrainStep ..> Netgraph : receive
  OutputTrainStep ..> Netgraph : r eceive
  LossTrainStep ..> Netgraph : receive
  BackwardTrainStep ..> Netgraph : receive
  Netgraph ..> InitStep : yield
  Netgraph ..> ForwardTrainStep : yield
  Netgraph ..> OutputTrainStep : yield
  Netgraph ..> LossTrainStep : yield
  Netgraph ..> BackwardTrainStep : yield
  ForwardStep ..> ForwardTrainStep: extend
  OutputStep ..> OutputTrainStep : extend

  class Filter {
    prefilter(StepType) bool
    filter(StepType) bool
    postfilter(StepType) bool
  }
  Netgraph --* Filter
  Network --* Filter
  class TrainingData {
    + (input expected)
  }
  class Trainer {
    train1(float[] input, float[] expected)
    train(TrainingData)
  }
  class LossFunction {
    eval(float[]) float
    derivative(float) float[]
  }
  LossTrainStep --> LossFunction
  Trainer --o LossFunction
  LossFunction ..> Layer : Output
  LossFunction ..> Output
  Trainer ..> TrainingData
  Trainer --* Filter
  Trainer --* Network
  Netgraph --* Trainer

  class Image {

  }
  Netgraph "*" --> Image : Produce
```

### Step-Centric View

The view of the StepResult objects and how they relate to each other and to the objects that send and recieve them.

```mermaid
classDiagram
  InitStep ..> Trainer : receive
  ForwardStep ..> Trainer : receive
  OutputStep ..> Trainer : reeive
  Trainer ..> InitStep : yield
  Trainer ..> ForwardTrainStep : yield
  Trainer ..> OutputTrainStep : yield
  Trainer ..> LossTrainStep : yield
  Trainer ..> BackwardTrainStep : yield
  InitStep ..> Netgraph : recieve
  ForwardTrainStep ..> Netgraph : receive
  OutputTraiStep ..> Netgraph : receive
  LossTrainStep ..> Netgraph : receive
  BackwardTrainStep ..> Netgraph : receive
  Netgraph ..> InitStep : yield
  Netgraph ..> ForwardTrainStep : yield
  Hetgraph ..> OutputTrainStep : yield
  Netgraph ..> LossTrainStep : yield
  Netgraph ..> BackwardTrainStep : yield
  ForwardStep ..> ForwardTrainStep: extend
  OutputStep ..> OutputTrainStep : extend
  InitStep ..> Netgraph : receive
  EvalStep ..> Netgraph : receive
  Netgraph --> InitStep : yield
  Netgraph --> EvalStep : yield
  class Step {
    + StepType type
  }
  class InitStep {
    + float[] weights
  }
  class EvalStep {
    + StepType type
    + float[] values
  }
  class ForwardStep {

  }
  class OutputStep {

  }
  class TrainStep {
    + float[] input
    + float[] expected
  }
  TrainStep --|> Step
  class ForwardTrainStep {

  }
  ForwardTrainStep --|> EvalStep
  ForwardTrainStep --|> TrainStep
  class OutputTrainStep {

  }
  OutputTrainStep --|> OutputStep
  OutputTrainStep --|> TrainStep
  class LossFunction {
    eval(float[]) float
    derivative(float) float[]
  }
  class LossTrainStep {
    + float loss
  }
  LossTrainStep --|> ForwardTrainStep
  LossTrainStep --> LossFunction
  class BackwardTrainStep {
    + float loss
    + float[] gradient
    + float[] adjustment
  }
  BackwardTrainStep --|> TrainStep
  InitStep --|> Step
  EvalStep --|> Step
  ForwardStep --|> EvalStep
  OutputStep --|> EvalStep
  EvalStep --> Layer
  TrainStep --> Layer
  class Filter {
    prefilter(StepType) bool
    filter(StepType) bool
    postfilter(StepType) bool
  }

  InitStep ..> Filter          : test
  ForwardStep ..> Filter       : test
  OutputStep ..> Filter        : test
  FowardTrainStep ..> Filter   : test
  OutputTrainStep ..> Filter   : test
  LossTrainStep ..> Filter     : test
  BaclwardTrainStep ..> Filter : test
```

### User View

Full set of user-level objects.

```mermaid
classDiagram

  class Builder {
    + specification
  }
  class Randomizer {

  }
  class Network {
    + str name
    eval(float[]) float[]
    nodes() Node[]
    real_nodes() Node[]
    edges() Edge[]
    edges_in(Node) Edge[]
    edges_out(Node) Edge[]
  }
  class Netgraph {
    eval(float[]) float[]
    train1(float[] input, float[] expected)
    train(TrainingData)
  }
  class Filter {
    prefilter(StepType) bool
    filter(StepType) bool
    postfilter(StepType) bool
  }

  class ActivationFunction {
    eval(float) float
    derivative(float) float
  }

  class Trainer {
    train1(float[] input, float[] expected)
    train(TrainingData)
  }
  class LossFunction {
    eval(float[]) float
    derivative(float) float[]
  }

  class TrainingData {
    + (input expected)
  }

  class InputData {
    float[] data
  }

  class OutputData {
    float[] data
  }
  InputData ..> Netgraph : Invoke
  InputData ..> Network : Invoke
  Netgraph ..> OutputData : Produce
  Network ..> OutputData : Produce

  class Image {

  }
  Netgraph "*" --> Image : Produce

  Builder "0..1" ..> Network : Inject
  Randomizer "0..1" ..> Network : Inject
  ActivationFunction "0..1" ..> Builder : Inject
  Filter "0..1" ..> Network : Inject
  Filter "0..1" ..> Network : Invoke
  Trainer "1" --> Network
  LossFunction "0..1" ..> Trainer : Inject
  Filter "0..1" ..> Trainer : Inject
  Filter "0..1" ..> Trainer : Invoke
  Netgraph "1" --> Network
  Filter "0..1" ..> Netgraph : Inject
  Filter "0..1" ..> Netgraph : Invoke
  Netgraph "0..1" --> Trainer
  TrainingData ..> Trainer : Invoke

  Netgraph --> Image : Produce
  InputData ..> Netgraph : Invoke
  InputData ..> Network : Invoke
  Netgraph ..> OutputData : Produce
  Network ..> OutputData : Produce
```
