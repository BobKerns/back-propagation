# Network + Netgraph

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
