# User View

Full set of user-level objects.

```mermaid
classDiagram

  class Builder {
    + specification
  }
  class Randomizer {

  }
  class HeEtAl {

  }
  HeEtAl --|> Randomizer
  namespace core {
    class Network {
        + str name
        + Layer input_layer
        + Layer output_layer
        + Node[] nodes
        + Node[] real_nodes
        + Edge[] edges
        __call__(float[]) float[]
    }
    class Netgraph {
        __call__(float[]) float[]
        __call__(TrainingData)
    }
    class Trainer {
        train1(float[] input, float[] expected)
        __call__(TrainingData)
    }
  }

  namespace filter_and_trace {
    class Filter {
        prefilter(StepType) bool
        filter(StepType) bool
        postfilter(StepType) bool
    }
    class OutputOnlyFilter {
    }
    class EveryNFilter {
        + StepType[] types
        + int n
    }
    class FilterChain {

    }
    class Trace {
        __call__(StepType, StepResult)
    }
    class PrintTrace {

    }
    class CollectTrace {

    }
    class NullTrace {

    }
  }
  OutputOnlyFilter --|> Filter
  EveryNFilter --|> Filter
  FilterChain --|> Filter
  FilterChain "*" --> Filter
  CollectTrace --|> Trace
  NullTrace --|> Trace
  PrintTrace --|> Trace
  Trace ..> Network : Inject
  Trace ..> Trainer : Inject
  Trace ..> Netgraph : Inject
  Network ..> Trace : Invoke
  Trainer ..> Trace : Invoke
  Netgraph ..> Trace : Invoke
  class ActivationFunction {
    __call__(float) float
    derivative(float) float
  }
  class ReLU {

  }
  ReLU --|> ActivationFunction

  namespace LossFunctions {
    class LossFunction {
        __call__(float[]) float
        derivative(float) float[]
    }
    class MeanSquaredError {

    }
  }
  MeanSquaredError --|> LossFunction

  class TrainingData {
    + (input, expected)
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
