# Overview of Trace and Development Guide

The Trace library is a lightweight, modular package designed to allow developers to experiment easily with generative optimization and integrate feedback-driven learning into their computational workflows.
The library has four modules within the `opto` top-level namespace:

1. `opto.trace` provides the infrastructure for converting executing Python code into symbolic directed acyclic graphs (DAGs). 
It defines two tracing primitives:
    - `trace.node`: Wraps Python objects, designating them as nodes within the computational graph.
    - `@trace.bundle`: Decorates Python methods/functions, marking them as operators within the graph.

Each primitive has a `trainable` flag. 
When set to `True`, these marked nodes and bundles become the trainable *parameters* of the workflow.
By using these primitives, developers can create a *traced workflow* represented as a DAG.
This DAG structure is automatically constructed at runtime, capturing both computational dependencies and trainable parameters, ready for optimization.

2. `opto.optimizers` has an abstract class `Optimizer` that defines algorithms that take computation DAGs and associated feedback objects as input, and output values for the trainable parameters.
These algorithms are analogous to gradient-based optimizers in PyTorch, but are typically implemented as generative optimization agents, leveraging LLMs to analyze feedback and propose parameter updates.
We provide implementations of several generative optimizers:
    - `OptoPrime`
    - `TextGrad`
    - `OPRO`

3. `opto.trainers` has the `AlgorithmBase` abstraction that orchestrates the overall training process.
Trainers manage data handling, tracing control, feedback collection, optimizer invocation, and iterating/stopping. Specifically, a trainer:
    - Controls data sampling (via `DataLoader`).
    - Determines when DAGs are constructed and when feedback (e.g. via `AutoGuide`) is collected .
    - Invokes `optimizers` for parameter updates, possibly repeatedly and manages the training loop.
    - Logs training progress.

Although `optimizers` handle lower-level optimization decisions, trainers under `AlgorithmBase` manage broader training logic and are designed to be compatible across various `optimizers`.
We provide implementations of common trainers: `MinibatchAlgorithm`(basic minibatch training) and `BeamSearch` (example of search-based training).

4. `opto.utils` has a collection of reusable helper functions and backend utilities, including abstraction for:
    - Large Language Models (LLMs)
    - Databases
    - Miscellaneous support tools.

Note: Some utilities might require installing optional depedencies.

## Concise Summary of Abstractions
  - `trace`: Infrastructure to construct symbolic computational DAGs
  - `optimizers`: Receive DAG and feedback, output parameter values.
  - `trainer`: Manages DAG construction, data sampling, feedback collection, optimizer invocation, and training workflow control.

## Common Workflow for Using Trace

1. Define a traceable workflow with `trace.node` and `@trace.bundle`, marking trainable parameters. 
2. Wrap this workflow into a `trace.Module` with `@trace.model`.
3. Define a dataloader (`DataLoader`) and feedback oracle (analogous to a loss function, using e.g. `AutoGuide`). 
4. Instantiate a trainer from `opto.trainers`, specifying the optimizer from `opto.optimizers` alongside the defined module above, dataloader, and feedback oracle.
5. Run the trainer. 

## Guidelines for Improving and Extending Trace
  - **New optimization agents**: Contribute to `opto.optimizers`, sub-class from the `Optimizer` abstraction.
  - **New learning algorithms**: Contribute to `opto.trainers` (and optionally `opto.optimizers` if necessary). Design new algorithms sub-classing `AlgorithmBase`, new dataloader under `DataLoader`, or new feedback oracle under `AutoGuide`. 
  - **Improving infrastructure**: Propose modifications to `opto.trace` to improve tracing capability, user experience, or additional functionality.
  - **Onboarding other utility tools**: Add helpful tools to `opto.utils` and update `setup.py` accordingly for optional dependencies.
