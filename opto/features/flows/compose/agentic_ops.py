import opto.trace as trace
from typing import Tuple, Union, get_type_hints, Any, Dict, List, Optional, Callable
from opto.utils.llm import AbstractModel, LLM
from opto.features.flows.types import MultiModalPayload, QueryModel, StructuredInput, StructuredOutput, \
    ForwardMixin
from opto.trainer.guide import Guide
import numpy as np
import contextvars

# =========== Mixin for Agentic Optimizer ===========

"""

@llm_call
def auto_correct(solution: InitialSolution) -> ImprovedSolution:
    ...

class InitialSolution:
    "This is the intiial solution of a coding problem"

class ImprovedSolution:
    name: str

a = auto_correct(solution)

class CodeOptimizer:
    def __init__(self):
      # self.auto_correct = Loop(auto_correct)
      # self.auto_correct = Loop(self.auto_correct_op)
      self.auto_improve = Loop()

    def auto_correct_op(self, solution):
        prompt = "\n improve this."
        llm_response = self.llm(prompt + solution)
        return llm_response

    def forward(self):
      next_phase = self.auto_correct(solution)
      return self.auto_improve(next_phase)

TODO:
1. Support different styles of calling/initializing loop. Need to cover all use cases.
"""


class StopCondition(ForwardMixin):
    """
    A stop condition for the loop. It can be a callable class instance or a function.
    A simple stop condition is not necessary to be a subclass of this class.

    Implement an init method to pass in extra info into the stop condition

    Example Usage:

    class MaxIterationsOrConverged:
        def __init__(self, max_iters=5, threshold=0.01):
            self.max_iters = max_iters
            self.threshold = threshold

        def __call__(self, param_history: List, result_history: List) -> bool:
            # Stop if we've done enough iterations
            if len(result_history) >= self.max_iters:
                return True

            # Stop if results have converged (example)
            if len(result_history) >= 2:
                diff = abs(result_history[-1] - result_history[-2])
                if diff < self.threshold:
                    return True

            return False
    """

    def forward(self, param_history: List, result_history: List) -> bool:
        """The Loop will call the stop condition with the param_history and result_history.
        The stop condition should return a boolean value.
        """
        raise NotImplementedError("Need to implement the forward function")


class Loop(ForwardMixin):

    def __init__(self, func: Callable[[Any], Any], stop_condition: Any = None):
        assert callable(func), "func must be a callable"

        self.stop_condition = stop_condition
        self.func = func
        if stop_condition is not None:
            self.check_stop_condition(stop_condition)

    def check_stop_condition(self, stop_condition: Any):
        # Check if it's a callable class (has __call__ method and is an instance)
        if not callable(stop_condition):
            raise TypeError("stop_condition must be callable")

        # Check if it's a class instance (not a function or method)
        # Functions/methods don't have __dict__ or they have __self__ for bound methods
        if not hasattr(stop_condition, '__dict__') and not hasattr(stop_condition, '__self__'):
            raise TypeError("stop_condition must be a callable class instance, not a function")

        # Get type hints from the __call__ method
        try:
            hints = get_type_hints(stop_condition.__call__)
            if 'return' in hints:
                # Optional: validate that return type annotation is bool
                assert hints['return'] == bool, \
                    f"stop_condition.__call__ must be annotated to return bool, got {hints['return']}"
        except AttributeError:
            pass  # If __call__ doesn't have type hints, skip validation

    def step(self, *current_params: Any) -> Tuple[Any, Dict[str, Any]]:
        """Override this method to define the loop logic. If `func` is passed in during init, this is not necessary."""
        if self.func is not None:
            result, info = self.func(*current_params)
            return result, info
        raise NotImplementedError("Must provide func during initialization or override step method")

    def forward(self, *args, max_try: int = 10, stop_condition: Any = None) -> Tuple[
        Any, List[Dict[str, Any]]]:
        """
        Execute the loop with the given initial parameters.

        Args:
            *args: Initial parameters to pass to the function
            max_try: Maximum number of iterations
            stop_condition: Optional stop condition to override the one from __init__

        Returns:
            Tuple of (final_params, result_history)
        """
        param_history = []
        result_history = []
        current_params = args

        # Use the stop_condition from forward() if provided, otherwise use the one from __init__
        active_stop_condition = stop_condition if stop_condition is not None else self.stop_condition

        # Check if the initial parameter is already good enough (only if stop_condition exists)
        if active_stop_condition is not None:
            should_stop = active_stop_condition(param_history, result_history)
            assert isinstance(should_stop, (bool, np.bool_)), \
                f"stop_condition must return a boolean value, got {type(should_stop)}"

            if should_stop:
                return current_params if len(current_params) > 1 else (
                    current_params[0] if current_params else None), result_history

        for iteration in range(max_try):
            # Track the parameters before calling step
            param_history.append(
                current_params if len(current_params) > 1 else (current_params[0] if current_params else None))

            # Update current_params using step method
            result, info = self.step(*current_params)

            # Normalize result to always be a tuple for consistency
            if not isinstance(result, tuple):
                current_params = (result,)
            else:
                current_params = result

            # Track the result from step
            result_history.append(info)

            # Check stop condition after each step (only if it exists)
            if active_stop_condition is not None:
                should_stop = active_stop_condition(param_history, result_history)
                assert isinstance(should_stop, (bool, np.bool_)), \
                    f"stop_condition must return a boolean value, got {type(should_stop)}"

                if should_stop:
                    break

        # Return unpacked if single param, tuple if multiple params
        final_result = current_params if len(current_params) > 1 else (current_params[0] if current_params else None)
        return final_result, result_history


"""
Usage patterns:

Check(should_optimize, value)
    .then(lambda v: Loop(optimize_step, stop_condition)())
    .or_else(skip_optimization)()

class ConditionalUpdateLoop(Loop):
    def step(self, param):
        return Check(needs_large_step, param)
            .then(large_update)
            .or_else(small_update)()

Check(validate_data, data).then(
    lambda d: Check(needs_preprocessing, d)
        .then(lambda: Loop(preprocess, QualityStop())())
        .or_else(process_directly)()
).or_else(reject)()
"""


class Check(ForwardMixin):
    """A DSL for conditional execution with fluent interface."""

    def __init__(self, condition_func, *args, **kwargs):
        """
        Initialize the Check with a condition function and its arguments.

        Args:
            condition_func: A callable that returns a truthy/falsy value
            *args: Positional arguments to pass to the condition function
            **kwargs: Keyword arguments to pass to the condition function
        """
        self.condition_func = condition_func
        self.args = args
        self.kwargs = kwargs
        self.condition_result = None
        self.condition_evaluated = False
        self.then_func = None
        self.then_args = None
        self.then_kwargs = None
        self.elif_branches = []  # List of (condition, func, args, kwargs) tuples
        self.else_func = None
        self.else_args = None
        self.else_kwargs = None
        self.do_func = None
        self.do_args = None
        self.do_kwargs = None

    def _evaluate_condition(self):
        """Lazily evaluate the condition function."""
        if not self.condition_evaluated:
            result = self.condition_func(*self.args, **self.kwargs)
            # Store both the truthiness and the actual return value
            self.condition_result = result
            self.condition_evaluated = True
        return self.condition_result

    def then(self, callback_func, *extra_args, **extra_kwargs):
        """
        Define the function to execute if the condition is truthy.

        Args:
            callback_func: The function to execute if condition is true
            *extra_args: Additional positional arguments for the callback
            **extra_kwargs: Additional keyword arguments for the callback
        """
        self.then_func = callback_func
        self.then_args = extra_args
        self.then_kwargs = extra_kwargs
        return self

    def elseif(self, condition_func, callback_func, *extra_args, **extra_kwargs):
        """
        Add an elif branch with its own condition and callback.

        Args:
            condition_func: The condition to check if previous conditions were false
            callback_func: The function to execute if this condition is true
            *extra_args: Additional positional arguments for the callback
            **extra_kwargs: Additional keyword arguments for the callback
        """
        self.elif_branches.append((condition_func, callback_func, extra_args, extra_kwargs))
        return self

    def or_else(self, callback_func, *extra_args, **extra_kwargs):
        """
        Define the function to execute if all conditions are falsy.

        Args:
            callback_func: The function to execute if all conditions are false
            *extra_args: Additional positional arguments for the callback
            **extra_kwargs: Additional keyword arguments for the callback
        """
        self.else_func = callback_func
        self.else_args = extra_args
        self.else_kwargs = extra_kwargs
        return self

    # Alternative names for or_else
    otherwise = or_else
    else_ = or_else

    def do(self, callback_func, *extra_args, **extra_kwargs):
        """
        Define a function to execute after all branches, regardless of condition.

        Args:
            callback_func: The function to always execute at the end
            *extra_args: Additional positional arguments for the callback
            **extra_kwargs: Additional keyword arguments for the callback
        """
        self.do_func = callback_func
        self.do_args = extra_args
        self.do_kwargs = extra_kwargs
        return self

    def forward(self):
        """
        Execute the appropriate callback based on the condition results.

        Returns:
            The return value of whichever callback was executed,
            or the do callback if no branch was executed.
        """
        condition_result = self._evaluate_condition()
        execution_result = None
        branch_executed = False

        # Check main condition
        if condition_result:
            if self.then_func:
                # Combine original args with then args, plus condition result
                all_args = self.args + self.then_args
                # If condition returned a non-boolean value, include it
                if condition_result is not True:
                    all_args = (condition_result,) + all_args
                all_kwargs = {**self.kwargs, **self.then_kwargs}
                execution_result = self.then_func(*all_args, **all_kwargs)
                branch_executed = True

        # Check elif branches if main condition was false
        if not branch_executed:
            for elif_condition, elif_func, elif_args, elif_kwargs in self.elif_branches:
                # Evaluate elif condition with original args
                elif_result = elif_condition(*self.args, **self.kwargs)
                if elif_result:
                    # Combine args similar to then branch
                    all_args = self.args + elif_args
                    if elif_result is not True:
                        all_args = (elif_result,) + all_args
                    all_kwargs = {**self.kwargs, **elif_kwargs}
                    execution_result = elif_func(*all_args, **all_kwargs)
                    branch_executed = True
                    break

        # Execute else if no condition was true
        if not branch_executed and self.else_func:
            # Combine original args with else args
            all_args = self.args + self.else_args
            all_kwargs = {**self.kwargs, **self.else_kwargs}
            execution_result = self.else_func(*all_args, **all_kwargs)
            branch_executed = True

        # Always execute do if defined
        if self.do_func:
            # Do gets the execution result as first arg if there was one
            do_args = self.do_args
            if execution_result is not None:
                do_args = (execution_result,) + do_args
            all_kwargs = {**self.kwargs, **self.do_kwargs}
            do_result = self.do_func(*do_args, **all_kwargs)
            # Return the execution result if there was one, otherwise the do result
            return execution_result if execution_result is not None else do_result

        return execution_result