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
from opto.trainer.guide import Guide
from opto.utils.llm import AbstractModel, LLM

from typing import Any, Callable, Dict, List, Tuple, Optional

from opto.features.flows.compose import Loop, ChatHistory, StopCondition, Check

"""
A few design that it must have:
1. **multi-turn conversation by default (memory management)**
2. Flexibility to do tool-use
3. has scaffolding functions people can call to design their custom optimizer

First an abstract agent with the features, then implement?

Idea: write it like you would use for VeriBench
1. bug fix loop
2. external reward loop

initial task prompt -> initial solution
initial solution -> improvement prompt -> improved solution -> improvement prompt -> improved solution

(inside flow) Loop (use a boolean reward function) to keep executing

Inherits this optimizer
and specify the main forward by just calling different scaffolding functions
"""


# Base class that provides scaffolding
class AgenticOptimizer(Optimizer):
    def __init__(
            self,
            parameters: List[ParameterNode],
            *args,
            propagator: Propagator = None,
            **kwargs
    ):
        pass


"""
Add a veribench optimizer here.
A code optimizer is general, can work with kernel, others

initial_user_message -> initial_code -> bug fix here
initial_code -> improvement_prompt -> improved_code -> improvement_prompt -> improved_code

initial_code 
 1. check if there is a bug. If there is, try 10 times to fix. return the code --> make it correct/compilable
 2. take an improvement step --> make it run faster
 
improve initial code, bug fix 10 times -> improve bug-fixed code, bug fix 10 times -> ...

optimizer = (improve initial code, bug fix 10 times)
priority_search

TODO:
1. Make sure guide can be deep-copied
2. Declare a node that's trainable, we can have an initializer. When the node is created, we call the initializer.

add an init function to the optimizer API

define a class of Initializer

init function of the optimizer can receive some arguments

standardize optimizer API: 
- objective, context

State of the worker: the worker decides what to put into the state
(each candidate has its own optimizer)
(receive a candidate) (update rule is stateless)

Flip the order between

Use initializer on node, and then use **projection** on the node
Optimizer only focuses on improvement

LLM-based initializer, LLM-based projection

=============
planning agent: plan -> coding agent: write the code  

"""

class CodeOptimizer(Optimizer):
    def __init__(
            self,
            parameters: List[ParameterNode],
            llm: AbstractModel = None,
            *args,
            propagator: Propagator = None,
            bug_judge: Guide = None,  # compiler()
            reward_function: Callable[[str], float] = None,
            max_bug_fix_tries: int = 5,
            max_optimization_tries: int = 10,
            chat_max_len: int = 25,
            **kwargs
    ):
        super().__init__(parameters, *args, propagator=propagator, **kwargs)

        self.llm = llm or LLM()

        assert len(parameters) == 1, "CodeOptimizer expects a single ParameterNode as input"
        self.init_code = parameters[0]

        # Initialize chat history
        self.chat_history = ChatHistory(max_round=chat_max_len, auto_summary=False)

        # Store environment checker and reward function
        self.bug_judge = bug_judge
        self.max_bug_fix_tries = max_bug_fix_tries
        self.max_optimization_tries = max_optimization_tries

        self.task_description = None
        self.initial_instruction = None

        # 2. Do a single improvement step

    def initial_context(self, task_description, initial_instruction):
        """
        This provides the history of how the initial code was produced
        """
        self.task_description = task_description
        self.initial_instruction = initial_instruction
        self.chat_history.add_system_message(self.task_description)
        self.chat_history.add(self.initial_instruction,'user')
        self.chat_history.add(self.init_code, 'assistant')
    
    def bug_fix_step(self, lean4_code: str, max_try: int = 5) -> str:
        """
        This function is used to self-correct the Lean 4 code.
        It will be called by the LLM when the code does not compile.
        """

        # apply heuristic fixes 
        lean4_code = self.remove_import_error(lean4_code)
        valid, error_details = self.bug_judge(lean4_code)

        if valid:
            return lean4_code

        temp_conv_hist = self.chat_history.copy()

        counter = 0
        while not valid and counter < max_try:
            # sometimes LLM will hallucinate import error, so we remove that import statement

            print(f"Attempt {counter+1}: Fixing compilation errors")

            detailed_error_message = self.concat_error_messages(error_details)

            temp_conv_hist.add(detailed_error_message + "\n\n" + f"Lean code compilation FAILED with {len(error_details)} errors. If a theorem keeps giving error, you can use := sorry to skip it. Please wrap your lean code in ```lean and ```",
                               "user")

            raw_program = self.llm(temp_conv_hist.get_messages(), verbose=False)

            lean4_code = self.simple_post_process(raw_program)
            lean4_code = self.remove_import_error(lean4_code)

            valid, error_details = self.bug_judge(lean4_code)

            if valid:
                print(f"Successfully fixed errors after {counter} attempts")
                # we add to the round
                self.chat_history = self.chat_history + temp_conv_hist[-1].remove_system_message()
                return lean4_code
            else:
                counter += 1
                temp_conv_hist.add_message(lean4_code, "assistant")

        return lean4_code

    def _step(self, verbose=False, *args, **kwargs) -> Dict[ParameterNode, Any]:
        """
        Each step, we perform bug fix for a few rounds, then do one improvement step
        We add everything to the chat history
        """
        lean4_code = self.bug_fix_step(self.init_code.data, max_try=self.max_bug_fix_tries)
        # do one step improvement
        lean4_code = self.improve_step(lean4_code)

        return lean4_code

    def _extract_code(self, response: str) -> str:
        """Extract code from markdown code blocks."""
        import re
        # Match python code blocks
        pattern = r'```python\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # If no code block found, return the response as-is
        return response.strip()
