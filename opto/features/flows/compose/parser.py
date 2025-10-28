from typing import Optional, Callable, Any, Dict, Literal, get_type_hints
from abc import ABC, abstractmethod
import inspect
import json
from opto.features.flows.types import StructuredInput, StructuredOutput, ForwardMixin


class PromptAdapter(ABC):
    """Base adapter for converting structured types to/from LLM messages"""

    @abstractmethod
    def format_system_prompt(
            self,
            task_description: str,
            input_type: type[StructuredInput],
            output_type: type[StructuredOutput]
    ) -> str:
        """Generate system prompt explaining the task"""
        pass

    @abstractmethod
    def format_input(self, input_data: StructuredInput) -> str:
        """Convert input instance to string for LLM"""
        pass

    @abstractmethod
    def parse_output(self, llm_response: str, output_type: type[StructuredOutput]) -> StructuredOutput:
        """Parse LLM response into output instance"""
        pass


class JSONAdapter(PromptAdapter):
    """Standard JSON-based communication"""

    def format_system_prompt(self, task_description: str, input_type, output_type) -> str:
        output_schema = json.dumps(output_type.model_json_schema(), indent=2)
        return f"""Task: {task_description}

Output Schema:
{output_schema}

Respond with valid JSON matching the schema above."""

    def format_input(self, input_data: StructuredInput) -> str:
        return input_data.model_dump_json(indent=2)

    def parse_output(self, llm_response: str, output_type) -> StructuredOutput:
        # Try direct JSON parsing
        try:
            return output_type.model_validate_json(llm_response)
        except Exception:
            # Fallback: extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', llm_response, re.DOTALL)
            if json_match:
                return output_type.model_validate_json(json_match.group(1))
            raise ValueError(f"Failed to parse JSON response: {llm_response[:200]}")


class MarkdownAdapter(PromptAdapter):
    """Innovative markdown-based format using YAML-style frontmatter + sections"""

    def format_system_prompt(self, task_description: str, input_type, output_type) -> str:
        # Build field descriptions
        output_fields = []
        for field_name, field_info in output_type.model_fields.items():
            field_type = field_info.annotation.__name__ if hasattr(field_info.annotation, '__name__') else str(
                field_info.annotation)
            desc = field_info.description or ""
            output_fields.append(f"- **{field_name}** (`{field_type}`): {desc}")

        fields_str = "\n".join(output_fields)

        return f"""# Task
{task_description}

# Output Format
Respond using markdown with YAML frontmatter for metadata and sections for complex content.

## Required Fields
{fields_str}

## Structure
```markdown
---
field_name: simple value
other_field: simple value
---

# SectionField (if complex content)
Complex content here...

# AnotherSectionField
More complex content...
```

Use frontmatter for simple fields (strings, numbers, booleans).
Use markdown sections (# FieldName) for complex content (long text, lists, nested structures)."""

    def format_input(self, input_data: StructuredInput) -> str:
        """Format input as markdown with frontmatter"""
        lines = ["---"]

        # Simple fields in frontmatter
        complex_fields = {}
        for field_name, value in input_data.model_dump().items():
            if isinstance(value, (str, int, float, bool, type(None))):
                if isinstance(value, str) and ('\n' in value or len(value) > 100):
                    complex_fields[field_name] = value
                else:
                    lines.append(f"{field_name}: {value}")
            else:
                complex_fields[field_name] = value

        lines.append("---")
        lines.append("")

        # Complex fields as sections
        for field_name, value in complex_fields.items():
            lines.append(f"# {field_name}")
            if isinstance(value, str):
                lines.append(value)
            else:
                lines.append(json.dumps(value, indent=2))
            lines.append("")

        return "\n".join(lines)

    def parse_output(self, llm_response: str, output_type) -> StructuredOutput:
        """Parse markdown frontmatter + sections into structured output"""
        import re

        result = {}

        # Extract frontmatter
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', llm_response, re.DOTALL)
        if frontmatter_match:
            frontmatter = frontmatter_match.group(1)
            remaining = llm_response[frontmatter_match.end():]

            # Parse YAML-style frontmatter
            for line in frontmatter.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()

                    # Type coercion
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    else:
                        try:
                            value = float(value)
                        except ValueError:
                            pass  # Keep as string

                    result[key] = value
        else:
            remaining = llm_response

        # Extract sections
        sections = re.split(r'^# (\w+)\s*$', remaining, flags=re.MULTILINE)
        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                field_name = sections[i].strip()
                content = sections[i + 1].strip()
                result[field_name] = content

        return output_type(**result)


class XMLAdapter(PromptAdapter):
    """XML-based format similar to Anthropic's style"""

    def format_system_prompt(self, task_description: str, input_type, output_type) -> str:
        # Build field descriptions
        output_fields = []
        for field_name, field_info in output_type.model_fields.items():
            field_type = field_info.annotation.__name__ if hasattr(field_info.annotation, '__name__') else str(
                field_info.annotation)
            desc = field_info.description or ""
            output_fields.append(f"  <{field_name} type='{field_type}'>{desc}</{field_name}>")

        fields_str = "\n".join(output_fields)

        return f"""Task: {task_description}

Output Format: Respond with XML structure containing these fields:
<output>
{fields_str}
</output>

Provide actual values, not descriptions."""

    def format_input(self, input_data: StructuredInput) -> str:
        """Convert to XML format"""
        lines = ["<input>"]
        for field_name, value in input_data.model_dump().items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value)
            lines.append(f"  <{field_name}>{value}</{field_name}>")
        lines.append("</input>")
        return "\n".join(lines)

    def parse_output(self, llm_response: str, output_type) -> StructuredOutput:
        """Parse XML response"""
        import re

        result = {}

        # Extract fields using regex (simple XML parsing)
        for field_name in output_type.model_fields.keys():
            pattern = f'<{field_name}>(.*?)</{field_name}>'
            match = re.search(pattern, llm_response, re.DOTALL)
            if match:
                value = match.group(1).strip()

                # Try JSON parsing for complex types
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    pass

                result[field_name] = value

        return output_type(**result)


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
    """Enhanced wrapper supporting multiple function patterns"""

    def __init__(
            self,
            func: Callable,
            llm,
            input_type: type[StructuredInput],
            output_type: type[StructuredOutput],
            has_preprocessing: bool,
            adapter: PromptAdapter
    ):
        self.func = func
        self.llm = llm
        self.input_type = input_type
        self.output_type = output_type
        self.has_preprocessing = has_preprocessing
        self.adapter = adapter

        # Store output_type in adapter if it's MarkdownAdapter (for user message generation)
        if isinstance(adapter, MarkdownAdapter):
            adapter.output_type = output_type

        # Copy metadata
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__module__ = func.__module__
        self.__annotations__ = func.__annotations__

    def __call__(
            self,
            input_data: StructuredInput,
            system_prompt: Optional[str] = None
    ) -> StructuredOutput:
        """Execute with automatic LLM invocation"""

        # Validate input type
        if not isinstance(input_data, self.input_type):
            raise TypeError(f"Expected {self.input_type}, got {type(input_data)}")

        # Execute preprocessing if function has implementation
        if self.has_preprocessing:
            result = self.func(input_data)

            # Check if function returned the output type class (Usage 2 & 3)
            if inspect.isclass(result) and issubclass(result, StructuredOutput):
                self.output_type = result
                # Update adapter's output_type
                if isinstance(self.adapter, MarkdownAdapter):
                    self.adapter.output_type = result
            elif isinstance(result, StructuredOutput):
                # Function did full processing, return directly
                return result

        # Build system prompt using adapter
        if system_prompt is None:
            task_description = self.func.__doc__ or "Process the input data"
            system_prompt = self.adapter.format_system_prompt(
                task_description,
                self.input_type,
                self.output_type
            )

        # Format input using adapter
        user_message = self.adapter.format_input(input_data)

        # Invoke LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        response = self.llm(messages=messages)

        # Parse output using adapter
        output_instance = self.adapter.parse_output(response, self.output_type)

        return output_instance


def llm_call(
        func: Callable = None,
        *,
        llm=None,
        adapter: Literal["json", "markdown", "xml"] = "markdown",
        type_hints: Optional[Dict[str, str]] = None,
        **kwargs
):
    """
    Enhanced decorator supporting three usage patterns.

    Args:
        func: Function to decorate
        llm: LLM instance
        adapter: Output format - "json", "markdown", or "xml"
        type_hints: Custom type hints for markdown adapter fields, e.g.,
            {"confidence": "must be a single float value between 0 and 1"}
    """

    # Create adapter instance
    adapter_map = {
        "json": JSONAdapter(),
        "markdown": MarkdownAdapter(type_hints=type_hints),
        "xml": XMLAdapter()
    }
    adapter_instance = adapter_map[adapter]

    def decorator(f: Callable):
        hints = get_type_hints(f)

        # Extract input type (first parameter)
        params = list(hints.items())
        input_type = None
        for param_name, param_type in params:
            if param_name != 'return':
                input_type = param_type
                break

        # Get output type from annotation
        output_type = hints.get('return')

        # Detect if function has implementation (Usage 2 & 3)
        source = inspect.getsource(f)
        has_implementation = not source.strip().endswith('...')

        # Validate types
        if input_type and not issubclass(input_type, StructuredInput):
            raise TypeError(f"Input type must inherit from StructuredInput")

        if output_type and not issubclass(output_type, StructuredOutput):
            raise TypeError(f"Output type must inherit from StructuredOutput")

        # Use default LLM if none provided
        actual_llm = llm if llm is not None else LLM()

        return StructuredLLMCallable(
            f,
            actual_llm,
            input_type,
            output_type,
            has_implementation,
            adapter_instance
        )

    # Handle decorator syntax
    if func is None:
        return decorator
    else:
        return decorator(func)
