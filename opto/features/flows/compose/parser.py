import opto.trace as trace
from typing import Tuple, Union, get_type_hints, Any, Dict, List, Optional, Callable
from opto.utils.llm import AbstractModel, LLM
from opto.features.flows.types import MultiModalPayload, QueryModel, StructuredInput, StructuredOutput, \
    ForwardMixin
from opto.trainer.guide import Guide
import numpy as np
import contextvars

# =========== Structured LLM Input/Output With Parsing ===========

"""
Usage:

@llm_call
def evaluate_person(person: Person) -> Preference:
    "Evaluate if a person matches our criteria"
    ...

person = Person(name="Alice", age=30, income=75000)
preference = evaluate_person(person)

TODO 2: add trace bundle and input/output conversion
"""


class StructuredLLMCallable:
    """
    Wrapper class that makes a decorated function callable and automatically invokes the LLM.
    """

    def __init__(self, func: ForwardMixin, llm, input_type, output_type):
        self.func = func
        self.llm = llm
        self.input_type = input_type
        self.output_type = output_type

        # Store schemas
        self.input_schema = input_type.model_json_schema() if input_type else None
        self.output_schema = output_type.model_json_schema() if output_type else None

        # Copy function metadata
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__module__ = func.__module__
        self.__annotations__ = func.__annotations__

    def __call__(self, input_data: StructuredInput, system_prompt: Optional[str] = None) -> StructuredOutput:
        """
        Automatically invoke the LLM with the input data.

        Args:
            input_data: Instance of StructuredInput
            system_prompt: Optional custom system prompt. If not provided, uses default.

        Returns:
            Instance of StructuredOutput
        """
        # Validate input type
        if not isinstance(input_data, self.input_type):
            raise TypeError(f"Expected input of type {self.input_type}, got {type(input_data)}")

        # Convert input to string representation for LLM
        input_str = str(input_data)

        # Get function docstring as task description
        func_doc = self.func.__doc__ or "Process the input data"

        # Build system prompt
        if system_prompt is None:
            output_fields = list(self.get_output_fields().keys())
            system_prompt = f"""You are a helpful assistant that performs the following task: {func_doc}

You will receive input data and must produce output in JSON format with the following fields: {output_fields}

Output description: {self.output_type.get_docstring() or 'Structured output'}

Always respond with valid JSON only, no additional text."""

        # Build user message with input data
        user_message = f"""{input_str}

Please respond with a JSON object containing the required output fields."""

        # Construct messages in the format expected by the LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        # Invoke LLM
        response = self.llm(messages=messages)

        # Parse response into StructuredOutput
        # Note: Assumes LLM returns JSON string
        try:
            output_instance = self.output_type.model_validate_json(response)
        except Exception:
            # Fallback: try parsing as dict
            import json
            try:
                output_dict = json.loads(response)
                output_instance = self.output_type(**output_dict)
            except Exception as e:
                raise ValueError(f"Failed to parse LLM response into {self.output_type}: {e}\nResponse: {response}")

        return output_instance

    def get_input_docstring(self) -> Optional[str]:
        return self.input_type.get_docstring() if self.input_type else None

    def get_output_docstring(self) -> Optional[str]:
        return self.output_type.get_docstring() if self.output_type else None

    def get_input_fields(self) -> Dict[str, Any]:
        return self.input_type.get_fields_info() if self.input_type else {}

    def get_output_fields(self) -> Dict[str, Any]:
        return self.output_type.get_fields_info() if self.output_type else {}

    def __repr__(self):
        return f"<StructuredLLMCallable {self.__name__}>"


def llm_call(func: Callable = None, *, llm=None, **kwargs):
    """
    Decorator that extracts input/output schemas from type-annotated functions
    and creates a callable that automatically invokes the LLM.

    Args:
        func: The function to decorate (automatically passed when used without arguments)
        llm: AbstractModel instance to use for LLM calls
        **kwargs: Additional LLM configuration parameters

    Usage:
        # Without arguments
        @llm_call
        def process_person(person: Person) -> Preference:
            '''Evaluate if a person matches our criteria'''
            pass

        # With arguments
        @llm_call(llm=customized_llm)
        def process_person(person: Person) -> Preference:
            '''Evaluate if a person matches our criteria'''
            pass

        # Call it directly - LLM is invoked automatically
        person = Person(name="Alice", age=30, income=75000)
        result = process_person(person)  # Returns Preference instance

        # Access schemas
        process_person.input_type
        process_person.output_type
        process_person.input_schema
        process_person.output_schema
    """

    def decorator(f: Callable):
        hints = get_type_hints(f)

        # Get first parameter type and return type
        params = list(hints.items())
        input_type = None
        output_type = None

        # Find first non-return parameter
        for param_name, param_type in params:
            if param_name != 'return':
                input_type = param_type
                break

        output_type = hints.get('return')

        # Validate types
        if input_type and not issubclass(input_type, StructuredInput):
            raise TypeError(f"Input type {input_type} must inherit from StructuredInput")

        if output_type and not issubclass(output_type, StructuredOutput):
            raise TypeError(f"Output type {output_type} must inherit from StructuredOutput")

        # Use default LLM if none provided
        # Note: Replace this with your actual default LLM initialization
        actual_llm = llm
        if actual_llm is None:
            actual_llm = LLM()  # we use the default LLM

        # Create and return the callable wrapper
        return StructuredLLMCallable(f, actual_llm, input_type, output_type)

    # Handle both @llm_call and @llm_call(...) syntax
    if func is None:
        # Called with arguments: @llm_call(llm=custom_llm)
        return decorator
    else:
        # Called without arguments: @llm_call
        return decorator(func)


# =========== </Structured LLM Input/Output With Parsing> ===========