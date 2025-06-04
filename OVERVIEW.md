# Overview of Trace and Development Guide

The library of Trace is designed to be a lightweight, modularized package to allow developers to easily try new ideas on generative optimization and integrate learning wtih their pipelines. 

Currently, the Trace library has three main modules collected under the `opto` top module. 

1. `opto.trace` provides the infrastructure for tracing computational workflows. It defines two primitives `trace.node` and `@trace.bundle`. They can be applied to Python objects and methods, respectively, which define the root nodes and operators of the directed acyclic graph (DAG) of computation. They both have a `trainable` flag. When set `True`, the wrapped objects are viewed as *parameters* of the computational worflow. Users can use `trace.node` and `@trace.bundle` to declare the data and computation that they wish to trace and/or adapt, and we call the resulting workflow defined by these two primitives a *traced* workflow. When running a traced workflow, a DAG will be automatiically created by Trace as a data structure, which will later be sent to optimizers in `opto.optimizers`for updates (upon calling `node.backward` with soem feedback).

2. `opto.optimizers` has a collection of generative optimization algorithms, whose API is defined by an abstract class `Optimizer`. Think them like gradient algorithms. Their job is to propose a new version of the parameters (i.e. those set with `trainable=True`) when receiving a computational graph (DAG) and the feedback given to the computed output. Typically, these algorithms can be viewed as an LLM agent, which makes calls to LLM to analyze the computational graph and the feedback, and to propose updates. In Trace library, we provide implementation of several popular optimizers, such `OptoPrime`, `TextGrad`, and `OPRO`.

3. `opto.trainers` are a collection of training algorithms (under the `AlgorithmBase` class) that use optimizers in `opto.optimizers` as subroutines to improve a given workflow following a feedback oracle constructed by datasets, interactive environments, etc. While `Optimizer` defines a low-level *optimization* API, `AlgorithmBase` defines a high-level *learning* API which standarizes the format of agent (by the `Module` class created by `@trace.model`), the data loader (by the `DataLoader` class), and the feedback oracle (by the `AutoGuide` class). With this common abstraction, we offer training algorithms, from the basic `MinibatchAlgorithm` which trains minibatches of samples to search algorithms like `BeamSearch`. The `AlgorithmBase` also handles logging of the training process. While there are overlapping between the functions of `Optimizer` and `AlgorithmBase`, the main distinction is that algorithms under `AlgorithmBase` are meta algorithms, as they should work for different optimizers in `opto.optimizers`.


4. `opto.utils` has a collection of helper functions and backends, which are reusable for various applications. This includes, e.g., abstraction of LLMs, database, etc. Making use of all these utils would requie installing optional depedencies.


In summary, `opto.trace` is the infrastructure, `opto.optimizers` are algorithms that process feedback and propose new parameter candidates, and `opto.trainers` are algorithms built on top of `opto.trace` and `opto.optimizers` to train learning agents.

## Common Workflow of Using Trace

1. Use `trace.node` and `@trace.bundle` to define the traceable workflow and its trainable parameter. 
2. Wrap the workflow as a `trace.Module` using `@trace.model`
3. Create a dataloader using `DataLoader` and define the feedback oracle (an analogy of loss function) using `AutoGuide`. 
4. Create a trainer from `opto.trainers` using optimizers from `opto.optimizers` and the above module, dataloader, and feedback oracle.
5. Run the trainer. 


## Common Workflow of Improving Trace
- **Developing new optimization agent** Contribute to `trace.optimizers` and design new algorithms under `Optimizer`
- **Developing new learning algorithms** Contribute to `trace.trainers` (and `trace.optimizers` when necessary). Design new algorithms under `AlgorithmBase`, new dataloader under `DataLoader`, or new feedback oracle under `AutoGuide`. 
- **Improving infrastructure**  Propose updates to change `opto.trace` (e.g., to improve UI, add new tracing, etc.)
- **Onboarding other utility tools** Add to `opto.utils` and update `setup.py` with optional requirements.