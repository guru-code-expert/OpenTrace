import opto.trace as trace
from typing import Tuple, Union, get_type_hints, Any, Dict, List, Optional, Callable
from opto.utils.llm import AbstractModel, LLM
from opto.features.flows.types import MultiModalPayload, QueryModel, StructuredInput, StructuredOutput, \
    ForwardMixin
from opto.trainer.guide import Guide
import numpy as np
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
    def __init__(self, max_round=25, auto_summary=False):
        """Initialize chat history for multi-turn conversation.

        Args:
            max_round: Maximum number of conversation rounds (user-assistant pairs) to keep in history
            auto_summary: Whether to automatically summarize old messages
        """
        self.messages: List[Dict[str, Any]] = []
        self.max_round = max_round
        self.auto_summary = auto_summary

    def __len__(self):
        return len(self.messages)

    def add_system_message(self, content: Union[trace.Node, str]):
        """Add or replace a system message at the beginning of the chat history.

        Args:
            content: The content of the system message
        """
        # Check if the first message is a system message
        if len(self.messages) > 0 and self.messages[0].get("role") == "system":
            print("Warning: Replacing existing system message.")
            self.messages[0] = {"role": "system", "content": content}
        else:
            # Insert system message at the beginning
            self.messages.insert(0, {"role": "system", "content": content})
            self._trim_history()

    def add_message(self, content: Union[trace.Node, str], role):
        """Alias for add"""
        return self.add(content, role)

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
        if "role" not in message or "content" not in message or type(message) is not dict:
            raise ValueError("Message must have 'role' and 'content' fields.")
        self.add(message["content"], message["role"])

    def __iter__(self):
        return iter(self.messages)

    def __getitem__(self, index):
        """Get a specific round or slice of rounds as a ChatHistory object.

        Args:
            index: Integer index or slice object

        Returns:
            ChatHistory: A new ChatHistory object containing the selected round(s).
            Each round includes [system_prompt, user_prompt, response] where system_prompt is
            included if it exists in the chat history.
        """
        # Get user and assistant message indices
        user_indices = [i for i, msg in enumerate(self.messages) if msg["role"] == "user"]
        assistant_indices = [i for i, msg in enumerate(self.messages) if msg["role"] == "assistant"]

        # Build rounds (user-assistant pairs that are complete)
        rounds = []
        for user_idx in user_indices:
            # Find corresponding assistant response
            assistant_msg = None
            for asst_idx in assistant_indices:
                if asst_idx > user_idx:
                    assistant_msg = self.messages[asst_idx]
                    break

            if assistant_msg:
                rounds.append([self.messages[user_idx], assistant_msg])

        # Create new ChatHistory object
        new_history = ChatHistory(max_round=self.max_round, auto_summary=self.auto_summary)

        # Handle slicing
        if isinstance(index, slice):
            selected_rounds = rounds[index]
            # Add system message if it exists
            if self.messages and self.messages[0].get("role") == "system":
                new_history.messages.append(self.messages[0].copy())
            # Add all selected rounds
            for round_msgs in selected_rounds:
                for msg in round_msgs:
                    new_history.messages.append({"role": msg["role"], "content": msg["content"]})
            return new_history

        # Handle single index (including negative indexing)
        round_msgs = rounds[index]  # This will handle negative indices and raise IndexError if out of bounds

        # Add system message if it exists
        if self.messages and self.messages[0].get("role") == "system":
            new_history.messages.append({"role": self.messages[0]["role"], "content": self.messages[0]["content"]})

        # Add the selected round
        for msg in round_msgs:
            new_history.messages.append({"role": msg["role"], "content": msg["content"]})

        return new_history

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

    def copy(self, include_system=True):
        """Create a deep copy of the chat history.

        Args:
            include_system: Whether to include the system message in the copy (default: True)

        Returns:
            ChatHistory: A new ChatHistory instance with the same messages
        """
        new_history = ChatHistory(max_round=self.max_round, auto_summary=self.auto_summary)
        for message in self.messages:
            # Skip system message if include_system is False
            if not include_system and message["role"] == "system":
                continue
            # Create a new dict to avoid reference issues
            new_message = {"role": message["role"], "content": message["content"]}
            new_history.messages.append(new_message)
        return new_history

    def remove_system_message(self):
        """Create a copy of the chat history without the system message.

        Returns:
            ChatHistory: A new ChatHistory instance without the system message
        """
        return self.copy(include_system=False)

    def __add__(self, other):
        """Merge two chat histories together.

        Args:
            other: Another ChatHistory instance

        Returns:
            ChatHistory: A new ChatHistory instance with messages from both histories

        Raises:
            TypeError: If other is not a ChatHistory instance
            ValueError: If both histories have system prompts
        """
        if not isinstance(other, ChatHistory):
            raise TypeError("Can only add ChatHistory instances together")

        # Check if both have system messages
        has_system_self = self.messages and self.messages[0].get("role") == "system"
        has_system_other = other.messages and other.messages[0].get("role") == "system"

        if has_system_self and has_system_other:
            raise ValueError("Cannot merge two chat histories that both have system prompts")

        # Create new history with max of the two max_rounds
        new_history = ChatHistory(
            max_round=max(self.max_round, other.max_round),
            auto_summary=self.auto_summary or other.auto_summary
        )

        # Add messages from self
        for message in self.messages:
            new_message = {"role": message["role"], "content": message["content"]}
            new_history.messages.append(new_message)

        # Add messages from other, skipping system message if self already has one
        for message in other.messages:
            if has_system_self and message["role"] == "system":
                continue
            new_message = {"role": message["role"], "content": message["content"]}
            new_history.messages.append(new_message)

        # Trim the combined history to respect max_round
        new_history._trim_history()

        return new_history

    def _trim_history(self):
        """Trim history to max_round while preserving system message and the first round."""
        # Count the number of rounds (user-assistant pairs)
        user_indices = [i for i, msg in enumerate(self.messages) if msg["role"] == "user"]
        assistant_indices = [i for i, msg in enumerate(self.messages) if msg["role"] == "assistant"]

        # If we don't have enough messages to form complete rounds, return
        if len(user_indices) <= self.max_round or len(assistant_indices) < len(user_indices):
            return

        protected_messages = []

        # Keep system message if it exists
        if self.messages and self.messages[0].get("role") == "system":
            protected_messages.append(self.messages[0])

        # Always keep the first round (first user message and first assistant response)
        if user_indices:
            first_user_idx = user_indices[0]
            protected_messages.append(self.messages[first_user_idx])

            # Find the first assistant message after the first user message
            for i in assistant_indices:
                if i > first_user_idx:
                    protected_messages.append(self.messages[i])
                    break

        # Calculate how many recent rounds we can keep
        num_protected_rounds = 1 if user_indices else 0  # We've protected the first round if it exists
        remaining_rounds = self.max_round - num_protected_rounds

        if remaining_rounds > 0:
            # Get the most recent rounds (user-assistant pairs)
            recent_rounds = []

            # Start from the most recent user message and work backwards
            for i in range(len(user_indices) - 1, num_protected_rounds - 1, -1):
                if len(recent_rounds) // 2 >= remaining_rounds:
                    break

                user_idx = user_indices[i]
                user_msg = self.messages[user_idx]

                # Find the corresponding assistant response
                assistant_msg = None
                for j in assistant_indices:
                    if j > user_idx:
                        assistant_msg = self.messages[j]
                        break

                if assistant_msg:
                    # Add the pair in chronological order
                    recent_rounds = [user_msg, assistant_msg] + recent_rounds

            # Combine protected messages with recent rounds
            self.messages = protected_messages + recent_rounds
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

    def chat(self, user_query: str, payload: Optional[MultiModalPayload] = None,
             chat_history_on: Optional[bool] = None) -> str:
        """Note that chat/forward always assumes it's a single turn of the conversation. History/context management will be accomplished
           through other APIs"""
        return self.forward(user_query, payload, chat_history_on)

# =========== </LLM Base Model> ===========
