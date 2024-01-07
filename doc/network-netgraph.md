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
      + is_bias bool
      + label str
      + gradient float[]
      + activation ActivationFunction
      + layer Layer
      + idx int
      + position Point
      + edges Edge[]
      + edges_in Edge[]
      + edges_out Edge[]
  }
  class Edge {
    + Node from_
    + Node to_
    + float weight
    + str label
  }
  class Layer {
    + int position
    + str label
    + LayerType type
    + float[] values
    + Node bias
    + node nodes
    + node[] real_nodes
    + int index
    + Point position
    add_node(Node)
    add_nodes(Nodes[])
  }
  class ActivationFunction {
    __call__(float) float
    derivative(float) float
  }
  class Network {
    + str name
    + Layer input_layer
    + Layer output_layer
    + Node[] nodes
    + Node[] real_nodes
    + Edge[] edges
    __call__(float[]) float[]
    with filter(Filter) Filter
    with step_active(Filter) filter
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
