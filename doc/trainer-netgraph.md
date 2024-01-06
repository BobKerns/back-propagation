# Trainer + Netgraph

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
