"""
An Agentic Optimizer that has access to tools and other resources to perform complex optimization tasks.
Particularly useful for formal theorem proving, code optimization (including kernel code), and algorithm designs.

We use types defined in opto.features.flows
"""

import json
from textwrap import dedent
from dataclasses import dataclass, asdict
from typing import Dict

from opto.trace.nodes import ParameterNode, Node, MessageNode
from opto.trace.propagators import TraceGraph, GraphPropagator
from opto.trace.propagators.propagators import Propagator

from opto.optimizers.optoprime_v2 import OptoPrimeV2, OptimizerPromptSymbolSet
from opto.optimizers.optimizer import Optimizer
from opto.utils.llm import AbstractModel, LLM

from typing import Any, List, Dict, Union, Tuple, Optional

"""
A few design that it must have:
1. multi-turn conversation by default (memory management)
2. can take in tools (RAG in particular, but MCP servers as well)

First an abstract agent with the features, then implement?

Idea: write it like you would use for VeriBench
1. bug fix loop
2. external reward loop

initial task prompt -> initial solution
initial solution -> improvement prompt -> improved solution -> improvement prompt -> improved solution
"""


class AgenticOptimizer(Optimizer):
    def __init__(
            self,
            parameters: List[ParameterNode],
            llm: AbstractModel = None,
            *args,
            propagator: Propagator = None,
            objective: Union[None, str] = None,
            ignore_extraction_error: bool = True,
            # ignore the type conversion error when extracting updated values from LLM's suggestion
            include_example=False,
            memory_size=0,  # Memory size to store the past feedback
            max_tokens=4096,
            log=True,
            initial_var_char_limit=2000,
            optimizer_prompt_symbol_set: OptimizerPromptSymbolSet = None,
            use_json_object_format=True,  # whether to use json object format for the response when calling LLM
            **kwargs,
    ):
        pass
