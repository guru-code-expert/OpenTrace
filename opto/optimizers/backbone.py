"""
Flexible conversation manager for multi-turn LLM conversations.
Uses LiteLLM unified format for all providers (OpenAI, Anthropic, Google, etc.).
"""

from typing import List, Dict, Any, Optional, Literal, Union
from dataclasses import dataclass, field
import json
import base64
from pathlib import Path
import warnings


@dataclass
class TextContent:
    """Text content block"""
    type: Literal["text"] = "text"
    text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "text": self.text}


@dataclass
class ImageContent:
    """Image content block - supports URLs or base64"""
    type: Literal["image"] = "image"
    image_url: Optional[str] = None
    image_data: Optional[str] = None  # base64 encoded
    media_type: str = "image/jpeg"  # image/jpeg, image/png, image/gif, image/webp
    detail: Optional[str] = None  # OpenAI: "auto", "low", "high"

    def to_dict(self) -> Dict[str, Any]:
        if self.image_url:
            return {
                "type": self.type,
                "image_url": self.image_url,
                "media_type": self.media_type
            }
        else:
            return {
                "type": self.type,
                "image_data": self.image_data,
                "media_type": self.media_type
            }

    @classmethod
    def from_file(cls, filepath: str, media_type: Optional[str] = None):
        """Load image from file"""
        path = Path(filepath)
        if not media_type:
            ext_to_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            media_type = ext_to_type.get(path.suffix.lower(), 'image/jpeg')

        with open(filepath, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        return cls(image_data=image_data, media_type=media_type)


@dataclass
class PDFContent:
    """PDF content block"""
    type: Literal["pdf"] = "pdf"
    pdf_url: Optional[str] = None
    pdf_data: Optional[str] = None  # base64 encoded
    filename: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        if self.pdf_url:
            return {
                "type": "document",
                "source": {"type": "url", "url": self.pdf_url},
                "filename": self.filename
            }
        else:
            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": self.pdf_data
                },
                "filename": self.filename
            }

    @classmethod
    def from_file(cls, filepath: str):
        """Load PDF from file"""
        path = Path(filepath)
        with open(filepath, 'rb') as f:
            pdf_data = base64.b64encode(f.read()).decode('utf-8')

        return cls(pdf_data=pdf_data, filename=path.name)


@dataclass
class FileContent:
    """Generic file content block (for code, data files, etc.)"""
    file_data: str  # Could be text content or base64 for binary
    filename: str
    type: Literal["file"] = "file"
    mime_type: str = "text/plain"
    is_binary: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "file_data": self.file_data,
            "is_binary": self.is_binary
        }

    @classmethod
    def from_file(cls, filepath: str, mime_type: Optional[str] = None):
        """Load file from disk"""
        path = Path(filepath)

        # Try to read as text first
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                file_data = f.read()
                is_binary = False
        except UnicodeDecodeError:
            # Fall back to binary
            with open(filepath, 'rb') as f:
                file_data = base64.b64encode(f.read()).decode('utf-8')
                is_binary = True

        if not mime_type:
            # Simple mime type detection
            ext_to_type = {
                '.py': 'text/x-python',
                '.js': 'text/javascript',
                '.json': 'application/json',
                '.csv': 'text/csv',
                '.txt': 'text/plain',
                '.md': 'text/markdown',
                '.html': 'text/html',
            }
            mime_type = ext_to_type.get(path.suffix.lower(), 'application/octet-stream')

        return cls(
            file_data=file_data,
            filename=path.name,
            mime_type=mime_type,
            is_binary=is_binary
        )


# Union type for all content types
ContentBlock = Union[TextContent, ImageContent, PDFContent, FileContent]


@dataclass
class ToolCall:
    """Represents a tool call made by the LLM"""
    id: str
    type: str  # "function", "web_search", etc.
    name: Optional[str] = None  # function name
    arguments: Optional[Dict[str, Any]] = None  # function arguments

    def to_dict(self) -> Dict[str, Any]:
        result = {"id": self.id, "type": self.type}
        if self.name:
            result["name"] = self.name
        if self.arguments:
            result["arguments"] = self.arguments
        return result


@dataclass
class ToolResult:
    """Represents the result of a tool execution"""
    tool_call_id: str
    content: str  # Result as string (can be JSON stringified)
    is_error: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "content": self.content,
            "is_error": self.is_error
        }


@dataclass
class ToolDefinition:
    """Defines a tool that the LLM can use"""
    type: str  # "function", "web_search", "file_search", etc.
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    strict: bool = False  # OpenAI strict mode
    # Provider-specific fields
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.type}
        if self.name:
            result["name"] = self.name
        if self.description:
            result["description"] = self.description
        if self.parameters:
            result["parameters"] = self.parameters
        if self.strict:
            result["strict"] = self.strict
        result.update(self.extra)
        return result


@dataclass
class UserTurn:
    """Represents a user message turn in the conversation"""
    content: List[ContentBlock] = field(default_factory=list)
    tools: List[ToolDefinition] = field(default_factory=list)

    # Provider-specific settings
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None

    # Metadata
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_text(self, text: str) -> 'UserTurn':
        """Add text content"""
        self.content.append(TextContent(text=text))
        return self

    def add_image(self, url: Optional[str] = None, data: Optional[str] = None,
                  media_type: str = "image/jpeg") -> 'UserTurn':
        """Add image content"""
        self.content.append(ImageContent(
            image_url=url,
            image_data=data,
            media_type=media_type
        ))
        return self

    def add_image_file(self, filepath: str) -> 'UserTurn':
        """Add image from file"""
        self.content.append(ImageContent.from_file(filepath))
        return self

    def add_pdf(self, url: Optional[str] = None, data: Optional[str] = None) -> 'UserTurn':
        """Add PDF content"""
        self.content.append(PDFContent(pdf_url=url, pdf_data=data))
        return self

    def add_pdf_file(self, filepath: str) -> 'UserTurn':
        """Add PDF from file"""
        self.content.append(PDFContent.from_file(filepath))
        return self

    def add_file(self, filepath: str, mime_type: Optional[str] = None) -> 'UserTurn':
        """Add file from disk"""
        self.content.append(FileContent.from_file(filepath, mime_type))
        return self

    def add_tool(self, tool: ToolDefinition) -> 'UserTurn':
        """Add a tool definition"""
        self.tools.append(tool)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "role": "user",
            "content": [c.to_dict() for c in self.content],
            "tools": [t.to_dict() for t in self.tools] if self.tools else None,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "metadata": self.metadata
        }

    def to_litellm_format(self) -> Dict[str, Any]:
        """Convert to LiteLLM format (OpenAI-compatible, works with all providers)"""
        content = []
        for block in self.content:
            if isinstance(block, TextContent):
                content.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageContent):
                if block.image_url:
                    img_dict = {"type": "image_url", "image_url": {"url": block.image_url}}
                    if block.detail:
                        img_dict["image_url"]["detail"] = block.detail
                    content.append(img_dict)
                else:
                    data_url = f"data:{block.media_type};base64,{block.image_data}"
                    content.append({"type": "image_url", "image_url": {"url": data_url}})
            elif isinstance(block, PDFContent):
                # LiteLLM supports PDFs for providers like Claude
                # Use image_url type with PDF data URL for compatibility
                if block.pdf_url:
                    warnings.warn("PDF URLs may not be supported by all providers through LiteLLM")
                    content.append({"type": "text", "text": f"[PDF: {block.pdf_url}]"})
                else:
                    # Encode as data URL for providers that support PDFs
                    data_url = f"data:application/pdf;base64,{block.pdf_data}"
                    content.append({"type": "image_url", "image_url": {"url": data_url}})
            elif isinstance(block, FileContent):
                # For file content, add as text or data URL based on type
                if block.is_binary:
                    data_url = f"data:{block.mime_type};base64,{block.file_data}"
                    content.append({"type": "text", "text": f"[File: {block.filename}]\n{data_url}"})
                else:
                    content.append({"type": "text", "text": f"[File: {block.filename}]\n{block.file_data}"})

        return {
            "role": "user",
            "content": content
        }


@dataclass
class AssistantTurn:
    """Represents an assistant message turn in the conversation"""
    content: List[ContentBlock] = field(default_factory=list)

    # Tool usage (Option B: Everything in AssistantTurn)
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)

    # Provider-specific features
    reasoning: Optional[str] = None  # OpenAI reasoning/thinking
    finish_reason: Optional[str] = None  # "stop", "length", "tool_calls", etc.

    # Token usage
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    # Metadata
    model: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_text(self, text: str) -> 'AssistantTurn':
        """Add text content"""
        self.content.append(TextContent(text=text))
        return self

    def add_image(self, url: Optional[str] = None, data: Optional[str] = None,
                  media_type: str = "image/jpeg") -> 'AssistantTurn':
        """Add image content (some models can generate images)"""
        self.content.append(ImageContent(
            image_url=url,
            image_data=data,
            media_type=media_type
        ))
        return self

    def add_tool_call(self, tool_call: ToolCall) -> 'AssistantTurn':
        """Add a tool call"""
        self.tool_calls.append(tool_call)
        return self

    def add_tool_result(self, result: ToolResult) -> 'AssistantTurn':
        """Add a tool result"""
        self.tool_results.append(result)
        return self

    def get_text(self) -> str:
        """Get all text content concatenated"""
        return " ".join(
            block.text for block in self.content
            if isinstance(block, TextContent)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "role": "assistant",
            "content": [c.to_dict() for c in self.content],
            "tool_calls": [tc.to_dict() for tc in self.tool_calls] if self.tool_calls else None,
            "tool_results": [tr.to_dict() for tr in self.tool_results] if self.tool_results else None,
            "reasoning": self.reasoning,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "model": self.model,
            "metadata": self.metadata
        }

    def to_litellm_format(self) -> Dict[str, Any]:
        """Convert to LiteLLM format (OpenAI-compatible, works with all providers)"""
        result = {"role": "assistant"}

        if self.content:
            # For multimodal or simple text response
            text = self.get_text()
            if text:
                result["content"] = text

        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments) if tc.arguments else "{}"
                    }
                }
                for tc in self.tool_calls
            ]

        return result


@dataclass
class ConversationHistory:
    """Manages conversation history across multiple turns using LiteLLM unified format"""
    turns: List[Union[UserTurn, AssistantTurn]] = field(default_factory=list)
    system_prompt: Optional[str] = None
    protected_rounds: int = 0  # Initial rounds to never truncate (task definition)

    def add_user_turn(self, turn: UserTurn) -> 'ConversationHistory':
        """Add a user turn"""
        self.turns.append(turn)
        return self

    def add_assistant_turn(self, turn: AssistantTurn) -> 'ConversationHistory':
        """Add an assistant turn"""
        self.turns.append(turn)
        return self

    def get_last_user_turn(self) -> Optional[UserTurn]:
        """Get the most recent user turn"""
        for turn in reversed(self.turns):
            if isinstance(turn, UserTurn):
                return turn
        return None

    def get_last_assistant_turn(self) -> Optional[AssistantTurn]:
        """Get the most recent assistant turn"""
        for turn in reversed(self.turns):
            if isinstance(turn, AssistantTurn):
                return turn
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "system_prompt": self.system_prompt,
            "protected_rounds": self.protected_rounds,
            "turns": [turn.to_dict() for turn in self.turns]
        }

    def to_litellm_format(
        self, 
        n: int = -1,
        truncate_strategy: Literal["from_start", "from_end"] = "from_start",
        protected_rounds: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Convert to LiteLLM messages format (OpenAI-compatible, works with all providers)
        
        Args:
            n: Number of historical rounds (user+assistant pairs) to include.
               -1 means all history (default: -1).
               The current (potentially incomplete) round is always included.
            truncate_strategy: How to truncate when n is specified:
                - "from_start": Remove oldest rounds, keep the most recent n rounds (default)
                - "from_end": Remove newest rounds, keep the oldest n rounds
            protected_rounds: Number of initial rounds to never truncate (task definition).
                If None, uses self.protected_rounds. These rounds count towards n, so
                if n=5 and protected_rounds=1, you get 1 protected + 4 truncatable rounds.
        
        Returns:
            List of message dictionaries in LiteLLM format
        """
        # Determine protected rounds
        n_protected = protected_rounds if protected_rounds is not None else self.protected_rounds
        protected_turns = n_protected * 2  # Each round = user + assistant
        
        # Apply truncation to turns
        if n == -1:
            selected_turns = self.turns
        else:
            # Protected rounds count towards N
            # So if N=5 and protected_rounds=1, we keep 1 protected + 4 from truncatable
            remaining_rounds = max(0, n - n_protected)
            
            # Split into protected and truncatable turns
            protected_part = self.turns[:protected_turns]
            truncatable_part = self.turns[protected_turns:]
            
            # remaining_rounds = number of rounds (pairs) from the truncatable part
            # Each round = 2 turns (user + assistant)
            # Plus include current incomplete round (if last turn is user, +1)
            has_incomplete_round = len(truncatable_part) > 0 and isinstance(truncatable_part[-1], UserTurn)
            n_turns = remaining_rounds * 2 + (1 if has_incomplete_round else 0)
            
            if truncate_strategy == "from_start":
                # Keep last n_turns from truncatable part (remove from start)
                truncated_part = truncatable_part[-n_turns:] if n_turns > 0 else []
            elif truncate_strategy == "from_end":
                # Keep first n_turns from truncatable part (remove from end)
                truncated_part = truncatable_part[:n_turns] if n_turns > 0 else []
            else:
                raise ValueError(f"Unknown truncate_strategy: {truncate_strategy}. Use 'from_start' or 'from_end'")
            
            # Combine protected + truncated
            selected_turns = protected_part + truncated_part
        
        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for turn in selected_turns:
            messages.append(turn.to_litellm_format())

            # Add tool results as separate messages in LiteLLM/OpenAI format
            if isinstance(turn, AssistantTurn) and turn.tool_results:
                for result in turn.tool_results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "content": result.content
                    })

        return messages
    
    def to_messages(
        self,
        n: int = -1,
        truncate_strategy: Literal["from_start", "from_end"] = "from_start",
        protected_rounds: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Alias for to_litellm_format() for convenience
        
        Args:
            n: Number of historical rounds (user+assistant pairs) to include.
               -1 means all history (default: -1).
               The current (potentially incomplete) round is always included.
            truncate_strategy: How to truncate when n is specified:
                - "from_start": Remove oldest rounds, keep the most recent n rounds (default)
                - "from_end": Remove newest rounds, keep the oldest n rounds
            protected_rounds: Number of initial rounds to never truncate (task definition).
                If None, uses self.protected_rounds. Counts towards n.
        
        Returns:
            List of message dictionaries in LiteLLM format
        """
        return self.to_litellm_format(n=n, truncate_strategy=truncate_strategy, protected_rounds=protected_rounds)

    def save_to_file(self, filepath: str):
        """Save conversation history to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'ConversationHistory':
        """Load conversation history from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # This is a simplified loader - you'd want more robust deserialization
        history = cls(
            system_prompt=data.get('system_prompt'),
            protected_rounds=data.get('protected_rounds', 0)
        )

        # Note: Full deserialization would require reconstructing objects from dicts
        # This is left as an exercise since it depends on your exact needs

        return history

    def clear(self):
        """Clear all turns from history"""
        self.turns.clear()

    def get_token_count_estimate(self) -> int:
        """Rough estimate of token count (actual count requires tokenizer)"""
        total = 0
        for turn in self.turns:
            if isinstance(turn, (UserTurn, AssistantTurn)):
                for block in turn.content:
                    if isinstance(block, TextContent):
                        # Very rough estimate: ~4 chars per token
                        total += len(block.text) // 4
        return total