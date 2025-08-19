---
hide:
  - navigation
---

# OpenTrace

**OpenTrace is an open-source, open-governance Python library for tracing and optimizing workflows using LLM-powered generative optimizers, maintained by the exact same group of developers for Trace.**

A typical LLM agent workflow is defined by a sequence of operations, which usually involve user-written Python **programs**, **instructions** to LLMs (e.g.,
prompts, few-shot examples, etc.), and LLM-generated programs to use external tools (e.g., Wikipedia, databases, Wolfram Alpha). Popular LLM libraries often focus on optimizing the instructions.
For example, libraries like LangChain focus on optimizing the LLM instructions by representing the instructions as special objects
and construct pre/post-processing functions to help users get the most out of LLM calls. In the example figure, this approach updates
and changes the brown squares of the agent workflow.

OpenTrace takes a different approach.
The user writes the Python program as usual, and then uses primitives like `node` and `@bundle` to wrap over their Python objects and functions and to designate which objects are trainable parameters.
This step is the **declare** phase where a user chooses how to represent the agent workflow as a graph.
After the user has declared the inputs and operations, OpenTrace captures the execution flow of the program as a graph. This step is the **forward** phase.
Finally, the user can optimize the entire program, such as by updating the LLM instructions, using OpenTrace. This step is the **optimize** phase.

<figure markdown="1" class="image-hover">
![Platform Overview](images/platform2.png){ width="50%" loading=lazy .shadow .zoom title="Platform Overview - Click to view larger" }
<figcaption>Platform Overview</figcaption>
</figure>

<style>
.shadow { 
  box-shadow: 0 4px 8px rgba(0,0,0,0.2); 
  border: 1px solid var(--md-default-fg-color--lightest);
}
.zoom { transition: all 0.3s ease; cursor: pointer; }
.zoom:hover { transform: scale(1.02); box-shadow: 0 8px 16px rgba(0,0,0,0.3); }
.image-hover { text-align: center; position: relative; }
.image-hover figcaption { 
  opacity: 0; 
  transition: opacity 0.3s ease; 
  margin-top: 6px; 
  font-size: 0.6em; 
  font-weight: bold;
  font-style: normal;
  color: var(--md-default-fg-color);
  background: white;
  border: 1px solid #ccc;
  box-shadow: 0 2px 4px rgba(0,0,0,0.2);
  padding: 4px 10px;
  border-radius: 3px;
  display: inline-block;
}
.image-hover:hover figcaption { opacity: 1; }
</style>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AgentOpt/Trace/blob/experimental/docs/examples/basic/greeting.ipynb)
[![GitHub](https://img.shields.io/github/stars/AgentOpt/OpenTrace?style=social)](https://github.com/AgentOpt/OpenTrace)

<div class="grid cards" markdown>

-   :material-graph-outline: **Execution Graph Tracing**

    ---

    Record traces of operations on Python objects and functions, automatically constructing execution graphs optimized for LLM workflows.

    [:octicons-arrow-right-24: Learn about tracing](#tracing-workflows)

-   :material-auto-fix: **LLM-Powered Optimization**

    ---

    Use generative optimizers to automatically improve your AI workflows end-to-end without manual prompt engineering.

    [:octicons-arrow-right-24: Explore optimizers](#optimization-system)

-   :material-code-braces: **PyTorch-Inspired API**

    ---

    Familiar gradient tape mechanism design reduces learning curve while providing powerful workflow optimization capabilities.

    [:octicons-arrow-right-24: See examples](#code-examples)

-   :material-puzzle-outline: **Framework Agnostic**

    ---

    Pure Python implementation with no API dependencies. Composable with any existing libraries and tools in your stack.

    [:octicons-arrow-right-24: Integration guide](#framework-integration)

</div>

---

## Installation

Get started with OpenTrace in just a few steps:

!!! tip "Quick Installation"
    ```bash
    pip install trace-opt
    ```

!!! info "Development Installation"
    For the latest features or to contribute:
    ```bash
    git clone https://github.com/AgentOpt/OpenTrace.git
    cd OpenTrace
    pip install -e .
    ```

!!! warning "Requirements"
    - Python 3.8+
    - No additional dependencies required for core functionality
    - Optional: OpenAI API key for LLM-powered optimization

[:material-download: Full Installation Guide](quickstart/installation.md){ .md-button .md-button--primary }

---

## Tracing Workflows

OpenTrace captures the execution flow of your Python programs as computational graphs, making it easy to understand and optimize complex AI workflows. Unlike traditional approaches that focus solely on prompt optimization, OpenTrace provides visibility into your entire pipeline.

!!! example "Key Features"
    - **Automatic graph construction** from Python execution
    - **Operation recording** for any Python objects and functions
    - **Execution flow visualization** for debugging and optimization
    - **Minimal overhead** with pure Python implementation

[:material-rocket-launch: Get Started with Tracing](quickstart/quick_start.ipynb){ .md-button .md-button--primary }

---

## Optimization System

OpenTrace uses LLM-powered generative optimizers to automatically improve your workflows. The system can optimize prompts, function implementations, and entire execution paths without manual intervention.

!!! tip "Optimization Capabilities"
    - **End-to-end optimization** of complete workflows
    - **Automatic prompt tuning** using feedback signals
    - **Code generation and refinement** for better performance
    - **Multi-step reasoning** improvement

[:material-auto-fix: Learn About Optimizers](tutorials/optimization_tutorial.ipynb){ .md-button .md-button--primary }

---

## Code Examples

OpenTrace features a PyTorch-inspired API design that makes it intuitive for developers familiar with gradient-based optimization. The familiar patterns reduce the learning curve while providing powerful capabilities.

!!! code-example "Quick Example"
    ```python
    from opto import trace
    
    @trace.model
    class MyAgent:
        def __init__(self):
            self.instruction = trace.node("Be helpful", trainable=True)
        
        def __call__(self, query):
            return trace.operators.call_llm(self.instruction, query)
    ```

[:material-code-braces: View All Examples](examples/basic/greeting.ipynb){ .md-button .md-button--primary }

---

## Framework Integration

OpenTrace is designed to be composable with existing tools and libraries. Its pure Python implementation means no external dependencies or API calls are required, making it easy to integrate into any workflow.

!!! integration "Compatibility"
    - **No external API dependencies** - works offline
    - **Composable design** - integrates with existing codebases  
    - **Flexible deployment** - works in any Python environment
    - **Library agnostic** - use with any ML/AI frameworks

[:material-puzzle-outline: Integration Examples](tutorials/basic_tutorial.ipynb){ .md-button .md-button--primary }