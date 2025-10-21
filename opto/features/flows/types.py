"""Types for opto flows."""
from typing import List, Dict, Union
from pydantic import BaseModel, model_validator
from typing import Any, Optional, Callable, Dict, Union, Type, List
from dataclasses import dataclass
import re
import json
from opto.optimizers.utils import encode_image_to_base64
from opto import trace


class TraceObject:
    def __str__(self):
        # Any subclass that inherits this will be friendly to the optimizer
        raise NotImplementedError("Subclasses must implement __str__")


class ForwardMixin:
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# ====== Multi-Modal LLM Support ======
class MultiModalPayload(BaseModel):
    image_bytes: Optional[str] = None  # base64-encoded data URL

    @classmethod
    def from_path(cls, path: str) -> "MultiModalPayload":
        """Create a payload by loading an image from a local file path."""
        data_url = encode_image_to_base64(path)
        return cls(image_bytes=data_url)

    def load_image(self, path: str) -> None:
        """Mutate the current payload to include a new image."""
        self.image_bytes = encode_image_to_base64(path)


class QueryModel(BaseModel):
    # Expose "query" as already-normalized: always a List[Dict[str, Any]]
    query: List[Dict[str, Any]]
    multimodal_payload: Optional[MultiModalPayload] = None

    @model_validator(mode="before")
    @classmethod
    def normalize(cls, data: Any):
        """
        Accepts:
          { "query": "hello" }
          { "query": "hello", "multimodal_payload": {"image_bytes": "..."} }
        And always produces:
          { "query": [ {text block}, maybe {image_url block} ], "multimodal_payload": ...}
        """
        if not isinstance(data, dict):
            raise TypeError("QueryModel input must be a dict")

        raw_query: Any = data.get("query")
        if isinstance(raw_query, trace.Node):
            assert isinstance(raw_query.data, (str, list)), "If using trace.Node, its data must be str"
            raw_query = raw_query.data

        # 1) Start with the text part
        if isinstance(raw_query, str):
            out: List[Dict[str, Any]] = [{"type": "text", "text": raw_query}]
        else:
            raise TypeError("`query` must be a string")

        # 2) If we have an image, append an image block
        payload = data.get("multimodal_payload")
        image_bytes: Optional[str] = None
        if payload is not None:
            if isinstance(payload, dict):
                image_bytes = payload.get("image_bytes")
            else:
                # Could be already-parsed MultiModalPayload
                image_bytes = getattr(payload, "image_bytes", None)

        if image_bytes:
            out = out + [{
                "type": "image_url",
                "image_url": {"url": image_bytes}
            }]

        # 3) Write back normalized fields
        data["query"] = out
        return data


# ====== </Multi-Modal LLM Support> ======

# ======= Structured LLM Info Support ======

from typing import Any, Dict, Optional, Type, get_type_hints
from pydantic import BaseModel, create_model, Field
import inspect


class StructuredData(BaseModel):
    """
    Base class for structured data (inputs/outputs) with support for both
    inheritance and dynamic on-the-fly usage.
    """

    _docstring: Optional[str] = None

    def __init_subclass__(cls, **kwargs):
        """Called when a class inherits from StructuredData"""
        super().__init_subclass__(**kwargs)
        cls._docstring = inspect.getdoc(cls)

    def __new__(cls, docstring: Optional[str] = None, **kwargs):
        """
        Handle both inheritance and on-the-fly usage.
        """
        # Check if being used dynamically (direct instantiation with docstring)
        if cls in (StructuredData, StructuredInput, StructuredOutput) and \
                docstring is not None and isinstance(docstring, str):
            # Determine the appropriate class name for dynamic instances
            if cls is StructuredInput or (cls is StructuredData and 'Input' in str(cls)):
                dynamic_name = 'DynamicStructuredInput'
            elif cls is StructuredOutput:
                dynamic_name = 'DynamicStructuredOutput'
            else:
                dynamic_name = 'DynamicStructuredData'

            dynamic_cls = type(
                dynamic_name,
                (cls,),
                {
                    '__doc__': docstring,
                    '_docstring': docstring,
                    '__module__': cls.__module__,
                }
            )
            instance = super(StructuredData, dynamic_cls).__new__(dynamic_cls)
            return instance

        return super().__new__(cls)

    def __init__(self, docstring: Optional[str] = None, **kwargs):
        """Initialize the instance"""
        dynamic_names = ('DynamicStructuredData', 'DynamicStructuredInput', 'DynamicStructuredOutput')
        if isinstance(docstring, str) and self.__class__.__name__ in dynamic_names:
            super().__init__(**kwargs)
            object.__setattr__(self, '_docstring', docstring)
        else:
            super().__init__(**kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        """Allow dynamic attribute setting for on-the-fly usage"""
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            dynamic_names = ('DynamicStructuredData', 'DynamicStructuredInput', 'DynamicStructuredOutput')
            if self.__class__.__name__ in dynamic_names:
                object.__setattr__(self, name, value)
            else:
                super().__setattr__(name, value)

    @classmethod
    def get_docstring(cls) -> Optional[str]:
        """Get the docstring of the class"""
        return getattr(cls, '_docstring', None) or inspect.getdoc(cls)

    @classmethod
    def get_fields_info(cls) -> Dict[str, Any]:
        """Get information about all fields"""
        base_names = ('StructuredData', 'StructuredInput', 'StructuredOutput')
        dynamic_names = ('DynamicStructuredData', 'DynamicStructuredInput', 'DynamicStructuredOutput')

        if cls.__name__ in dynamic_names or cls.__name__ in base_names:
            return {}

        fields_info = {}
        for field_name, field in cls.model_fields.items():
            fields_info[field_name] = {
                'type': field.annotation,
                'required': field.is_required(),
                'default': field.default if field.default is not None else None,
            }
        return fields_info

    def get_instance_fields(self) -> Dict[str, Any]:
        """Get all fields and their values from the instance"""
        dynamic_names = ('DynamicStructuredData', 'DynamicStructuredInput', 'DynamicStructuredOutput')
        if self.__class__.__name__ in dynamic_names:
            return {
                k: v for k, v in self.__dict__.items()
                if not k.startswith('_')
            }
        else:
            return self.model_dump()

    def __str__(self, template: Optional[str] = None) -> str:
        """
        Convert the structured data to a string format for LLM consumption.
        Subclasses can override for specific formatting.
        """
        docstring = self.get_docstring() or "No description provided"
        fields = self.get_instance_fields()

        if template:
            return template.format(docstring=docstring, fields=fields)

        lines = [f"Description: {docstring}", "", "Fields:"]
        for field_name, field_value in fields.items():
            lines.append(f"- {field_name}: {field_value}")

        return "\n".join(lines)


class StructuredInput(StructuredData):
    """
    Base class for structured inputs that can be used via inheritance or dynamically.
    """

    def __str__(self, template: Optional[str] = None) -> str:
        """
        Convert the structured input to a string format emphasizing input data.
        """
        docstring = self.get_docstring() or "No description provided"
        fields = self.get_instance_fields()

        if template:
            return template.format(docstring=docstring, fields=fields)

        lines = [f"Input: {docstring}", "", "Provided data:"]
        for field_name, field_value in fields.items():
            lines.append(f"- {field_name}: {field_value}")

        return "\n".join(lines)


class StructuredOutput(StructuredData):
    """
    Base class for structured outputs from LLM functions.
    """

    def __str__(self, template: Optional[str] = None) -> str:
        """
        Convert the structured output to a string format emphasizing results.
        """
        docstring = self.get_docstring() or "No description provided"
        fields = self.get_instance_fields()

        if template:
            return template.format(docstring=docstring, fields=fields)

        lines = [f"Output: {docstring}", "", "Results:"]
        for field_name, field_value in fields.items():
            lines.append(f"- {field_name}: {field_value}")

        return "\n".join(lines)

# ======= </Structured LLM Info Support> ======

# ======= Agentic Optimizer Support =======

# ======= </Agentic Optimizer Support> =======
