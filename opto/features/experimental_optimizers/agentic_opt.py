"""
An Agentic Optimizer that has access to tools and other resources to perform complex optimization tasks.
Particularly useful for formal theorem proving, code optimization (including kernel code), and algorithm designs.
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

from typing import Any, List, Dict, Union, Tuple, Optional

"""
A few design that it must have:
1. multi-turn conversation by default
2. can take in tools
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
            initial_var_char_limit=100,
            optimizer_prompt_symbol_set: OptimizerPromptSymbolSet = None,
            use_json_object_format=True,  # whether to use json object format for the response when calling LLM
            truncate_expression=truncate_expression,
            **kwargs,
    ):
        pass
