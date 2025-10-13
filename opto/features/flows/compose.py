import opto.trace as trace
from typing import Union, get_type_hints, Any, Dict, List, Optional, Callable
from opto.utils.llm import AbstractModel, LLM
from opto.features.flows.types import MultiModalPayload, QueryModel, StructuredInput, StructuredOutput
import contextvars

# =========== LLM Base Model ===========
"""
TracedLLM:
1. special operations that supports specifying inputs (system_prompt, user_prompt) to LLM and parsing of outputs, wrap
   everything under one command.
2. Easy to use interface -- can be inherited by users.
3. Support multi-turn chatting (message history)

Usage patterns:

Direct use: (only supports single input, single output) (signature: str -> str)
llm = TracedLLM("You are a helpful assistant.")
response = llm("Hello, what's the weather in France today?")
"""

USED_TracedLLM = contextvars.ContextVar('USED_TracedLLM', default=list())


class ChatHistory:
    def __init__(self, max_len=50, auto_summary=False):
        """Initialize chat history for multi-turn conversation.

        Args:
            max_len: Maximum number of messages to keep in history
            auto_summary: Whether to automatically summarize old messages
        """
        self.messages: List[Dict[str, Any]] = []
        self.max_len = max_len
        self.auto_summary = auto_summary

    def __len__(self):
        return len(self.messages)

    def add(self, content: Union[trace.Node, str], role):
        """Add a message to history with role validation.

        Args:
            content: The content of the message
            role: The role of the message ("user" or "assistant")
        """
        if role not in ["user", "assistant"]:
            raise ValueError(f"Invalid role '{role}'. Must be 'user' or 'assistant'.")

        # Check for alternating user/assistant pattern
        if len(self.messages) > 0:
            last_msg = self.messages[-1]
            if last_msg["role"] == role:
                print(f"Warning: Adding consecutive {role} messages. Consider alternating user/assistant messages.")

        self.messages.append({"role": role, "content": content})
        self._trim_history()

    def append(self, message: Dict[str, Any]):
        """Append a message directly to history."""
        if "role" not in message or "content" not in message:
            raise ValueError("Message must have 'role' and 'content' fields.")
        self.add(message["content"], message["role"])

    def __iter__(self):
        return iter(self.messages)

    def get_messages(self) -> List[Dict[str, str]]:
        messages = []
        for message in self.messages:
            if isinstance(message['content'], trace.Node):
                messages.append({"role": message["role"], "content": message["content"].data})
            else:
                messages.append(message)
        return messages

    def get_messages_as_node(self, llm_name="") -> List[trace.Node]:
        node_list = []
        for message in self.messages:
            # If user query is a node and has other computation attached, we can't rename it
            if isinstance(message['content'], trace.Node):
                node_list.append(message['content'])
            else:
                role = message["role"]
                content = message["content"]
                name = f"{llm_name}_{role}" if llm_name else f"{role}"
                if role == 'user':
                    name += "_query"
                elif role == 'assistant':
                    name += "_response"
                node_list.append(trace.node(content, name=name))

        return node_list

    def _trim_history(self):
        """Trim history to max_len while preserving first user message."""
        if len(self.messages) <= self.max_len:
            return

        # Find first user message index
        first_user_idx = None
        for i, msg in enumerate(self.messages):
            if msg["role"] == "user":
                first_user_idx = i
                break

        # Keep first user message
        protected_messages = []
        if first_user_idx is not None:
            first_user_msg = self.messages[first_user_idx]
            protected_messages.append(first_user_msg)

        # Calculate how many recent messages we can keep
        remaining_slots = self.max_len - len(protected_messages)
        if remaining_slots > 0:
            # Get recent messages
            recent_messages = self.messages[-remaining_slots:]
            # Avoid duplicating first user message
            if first_user_idx is not None:
                first_user_msg = self.messages[first_user_idx]
                recent_messages = [msg for msg in recent_messages if msg != first_user_msg]

            self.messages = protected_messages + recent_messages
        else:
            self.messages = protected_messages


DEFAULT_SYSTEM_PROMPT_DESCRIPTION = ("the system prompt to the agent. By tuning this prompt, we can control the "
                                     "behavior of the agent. For example, it can be used to provide instructions to "
                                     "the agent (such as how to reason about the problem, how to use tools, "
                                     "how to answer the question), or provide in-context examples of how to solve the "
                                     "problem.")


@trace.model
class TracedLLM:
    """
    This high-level model provides an easy-to-use interface for LLM calls with system prompts and optional chat history.

    Python usage patterns:

        llm = UF_LLM(system_prompt)
        response = llm.chat(user_prompt)
        response_2 = llm.chat(user_prompt_2)

    The underlying Trace Graph:
        TracedLLM_response0 = TracedLLM.forward.call_llm(args_0=system_prompt0, args_1=TracedLLM0_user_query0)
        TracedLLM_response1 = TracedLLM.forward.call_llm(args_0=system_prompt0, args_1=TracedLLM0_user_query0, args_2=TracedLLM_response0, args_3=TracedLLM0_user_query1)
        TracedLLM_response2 = TracedLLM.forward.call_llm(args_0=system_prompt0, args_1=TracedLLM0_user_query0, args_2=TracedLLM_response0, args_3=TracedLLM0_user_query1, args_4=TracedLLM_response1, args_5=TracedLLM0_user_query2)
    """

    def __init__(self,
                 system_prompt: Union[str, None, trace.Node] = None,
                 llm: AbstractModel = None, chat_history_on=False,
                 trainable=False, model_name=None):
        """Initialize TracedLLM with a system prompt.

        Args:
            system_prompt: The system prompt to use for LLM calls. If None and the class has a docstring, the docstring will be used.
            llm: The LLM model to use for inference
            chat_history_on: if on, maintain chat history for multi-turn conversations
            model_name: override the default name of the model
        """
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."

        self.system_prompt = trace.node(system_prompt, name='system_prompt',
                                        description=DEFAULT_SYSTEM_PROMPT_DESCRIPTION,
                                        trainable=trainable)
        # if system_prompt is already a node, then we have to override its trainable attribute
        self.system_prompt.trainable = trainable

        if llm is None:
            llm = LLM()
        assert isinstance(llm, AbstractModel), f"{llm} must be an instance of AbstractModel"
        self.llm = llm
        self.chat_history = ChatHistory()
        self.chat_history_on = chat_history_on

        current_llm_sessions = USED_TracedLLM.get()
        self.model_name = model_name if model_name else f"{self.__class__.__name__}{len(current_llm_sessions)}"
        current_llm_sessions.append(1)  # just a marker

    def forward(self, user_query: str,
                payload: Optional[MultiModalPayload] = None,
                chat_history_on: Optional[bool] = None) -> str:
        """This function takes user_query as input, and returns the response from the LLM, with the system prompt prepended.
        This method will always save chat history.

        If chat_history_on is set to False, the chat history will not be included in the LLM input.
        If chat_history_on is None, it will use the class-level chat_history_on setting.
        If chat_history_on is True, the chat history will be included in the LLM input.

        Args:
            user_query: The user query to send to the LLM. Can be

        Returns:
            str: For direct pattern
        """
        chat_history_on = self.chat_history_on if chat_history_on is None else chat_history_on

        user_message = QueryModel(query=user_query, multimodal_payload=payload).query

        messages = [{"role": "system", "content": self.system_prompt.data}]
        if chat_history_on:
            messages.extend(self.chat_history.get_messages())
        messages.append({"role": "user", "content": user_message})

        response = self.llm(messages=messages)

        @trace.bundle(output_name=f"{self.model_name}_response")
        def call_llm(*messages) -> str:
            """Call the LLM model.
            Args:
                messages: All the conversation history so far, starting from system prompt, to alternating user/assistant messages, ending with the current user query.
            Returns:
                response from the LLM
            """
            return response.choices[0].message.content

        user_query_node = trace.node(user_query, name=f"{self.model_name}_user_query")
        arg_list = [self.system_prompt]
        if chat_history_on:
            arg_list += self.chat_history.get_messages_as_node(self.model_name)
        arg_list += [user_query_node]

        response_node = call_llm(*arg_list)

        # save to chat history
        self.chat_history.add(user_query_node, role="user")
        self.chat_history.add(response_node, role="assistant")

        return response_node

    def chat(self, user_query: str, payload: Optional[MultiModalPayload] = None, chat_history_on: Optional[bool] = None) -> str:
        """Note that chat/forward always assumes it's a single turn of the conversation. History/context management will be accomplished
           through other APIs"""
        return self.forward(user_query, payload, chat_history_on)

# =========== </LLM Base Model> ===========

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

    def __init__(self, func: Callable, llm, input_type, output_type):
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
        return f"<LLMCallable {self.__name__}>"


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
            actual_llm = LLM() # we use the default LLM

        # Create and return the callable wrapper
        return StructuredLLMCallable(f, actual_llm, input_type, output_type)

    # Handle both @llm_call and @llm_call(...) syntax
    if func is None:
        # Called with arguments: @llm_call(llm=custom_llm)
        return decorator
    else:
        # Called without arguments: @llm_call
        return decorator(func)