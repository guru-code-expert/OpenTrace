"""
This adapater is for three cases:
1. OpenAI's response API (which is new and not fully supported by LiteLLM yet)
2. Google's MultiPart API design (not supported by LiteLLM response API at all)
3. Generic fallback option for all other providers (through LiteLLM)
"""

from typing import List, Tuple, Dict, Any, Callable, Union
import os
import time
import json
import os
import warnings
from .auto_retry import retry_with_exponential_backoff

import openai
from google import genai
from google.genai import types

try:
    import autogen  # We import autogen here to avoid the need of installing autogen
except ImportError:
    pass

class AbstractModel:
    """Abstract base class for LLM model wrappers with automatic refreshing.

    Provides a minimal abstraction for model APIs that need periodic refreshing
    for certificate renewal or memory management in long-running processes.

    Parameters
    ----------
    factory : callable
        A function that takes no arguments and returns a callable model instance.
    reset_freq : int or None, optional
        Number of seconds after which to refresh the model. If None, the model
        is never refreshed.

    Attributes
    ----------
    factory : callable
        The factory function for creating model instances.
    reset_freq : int or None
        Refresh frequency in seconds.
    model : Any
        Property that returns the current model instance.

    Methods
    -------
    __call__(*args, **kwargs)
        Execute the model, refreshing if needed.

    Notes
    -----
    This class handles:
    1. **Automatic Refreshing**: Recreates the model instance periodically
       to prevent issues with long-running connections.
    2. **Serialization**: Supports pickling by recreating the model on load.
    3. **Consistent Interface**: Ensures responses are available at
       `response['choices'][0]['message']['content']`.

    Subclasses should override the `model` property to customize behavior.

    See Also
    --------
    AutoGenLLM : Concrete implementation using AutoGen
    LiteLLM : Concrete implementation using LiteLLM
    """

    def __init__(self, factory: Callable, reset_freq: Union[int, None] = None) -> None:
        """
        Args:
            factory: A function that takes no arguments and returns a model that is callable.
            reset_freq: The number of seconds after which the model should be
                refreshed. If None, the model is never refreshed.
        """
        self.factory = factory
        self._model = self.factory()
        self.reset_freq = reset_freq
        self._init_time = time.time()

    # Overwrite this `model` property when subclassing.
    @property
    def model(self):
        """When self.model is called, text responses should always be available at `response['choices'][0]['message']['content']`"""
        return self._model

    # This is the main API
    def __call__(self, *args, **kwargs) -> Any:
        """ The call function handles refreshing the model if needed.
        """
        if self.reset_freq is not None and time.time() - self._init_time > self.reset_freq:
            self._model = self.factory()
            self._init_time = time.time()
        return self.model(*args, **kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._model = self.factory()


class AutoGenLLM(AbstractModel):
    """LLM wrapper using AutoGen's OpenAIWrapper for model interactions.

    This class provides integration with AutoGen for accessing various LLM APIs.
    It handles configuration, caching, and provides a consistent interface for
    the Trace framework.

    Parameters
    ----------
    config_list : list, optional
        List of model configurations. If None, attempts to load from
        'OAI_CONFIG_LIST' environment variable or auto-constructs from
        individual API keys.
    filter_dict : dict, optional
        Dictionary to filter configurations based on model properties.
    reset_freq : int or None, optional
        Number of seconds after which to refresh the model connection.

    Methods
    -------
    create(**config)
        Make a completion request with the given configuration.
    _factory(config_list)
        Class method to create the underlying AutoGen wrapper.

    Notes
    -----
    Configuration sources (in priority order):
    1. Explicitly provided config_list
    2. OAI_CONFIG_LIST environment variable or file
    3. Auto-construction from individual API keys (OPENAI_API_KEY, etc.)

    The create() method supports AutoGen's full configuration options including:
    - Templating with context
    - Response caching
    - Custom filter functions
    - API version specification

    For models not supported by AutoGen, subclass and override _factory()
    and create() methods.

    See Also
    --------
    AbstractModel : Base class for model wrappers
    auto_construct_oai_config_list_from_env : Helper for config construction

    Examples
    --------
    >>> # Using with explicit configuration
    >>> llm = AutoGenLLM(config_list=[{"model": "gpt-4", "api_key": "..."}])
    >>> 
    >>> # Using with environment variables
    >>> llm = AutoGenLLM()  # Auto-loads from environment
    >>> 
    >>> # Making a completion
    >>> response = llm(messages=[{"role": "user", "content": "Hello"}])
    """

    def __init__(self, config_list: List = None, filter_dict: Dict = None, reset_freq: Union[int, None] = None) -> None:
        if config_list is None:
            try:
                config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")
            except:
                config_list = auto_construct_oai_config_list_from_env()
                if len(config_list) > 0:
                    os.environ.update({"OAI_CONFIG_LIST": json.dumps(config_list)})
                config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")
        if filter_dict is not None:
            config_list = autogen.filter_config(config_list, filter_dict)

        factory = lambda *args, **kwargs: self._factory(config_list)
        super().__init__(factory, reset_freq)

    @classmethod
    def _factory(cls, config_list):
        return autogen.OpenAIWrapper(config_list=config_list)

    @property
    def model(self):
        return lambda **kwargs: self.create(**kwargs)

    # This is main API. We use the API of autogen's OpenAIWrapper
    def create(self, **config: Any):
        """Make a completion for a given config using available clients.
        Besides the kwargs allowed in openai's [or other] client, we allow the following additional kwargs.
        The config in each client will be overridden by the config.

        Args:
            - context (Dict | None): The context to instantiate the prompt or messages. Default to None.
                It needs to contain keys that are used by the prompt template or the filter function.
                E.g., `prompt="Complete the following sentence: {prefix}, context={"prefix": "Today I feel"}`.
                The actual prompt will be:
                "Complete the following sentence: Today I feel".
                More examples can be found at [templating](/docs/Use-Cases/enhanced_inference#templating).
            - cache (AbstractCache | None): A Cache object to use for response cache. Default to None.
                Note that the cache argument overrides the legacy cache_seed argument: if this argument is provided,
                then the cache_seed argument is ignored. If this argument is not provided or None,
                then the cache_seed argument is used.
            - agent (AbstractAgent | None): The object responsible for creating a completion if an agent.
            - (Legacy) cache_seed (int | None) for using the DiskCache. Default to 41.
                An integer cache_seed is useful when implementing "controlled randomness" for the completion.
                None for no caching.
                Note: this is a legacy argument. It is only used when the cache argument is not provided.
            - filter_func (Callable | None): A function that takes in the context and the response
                and returns a boolean to indicate whether the response is valid. E.g.,
            - allow_format_str_template (bool | None): Whether to allow format string template in the config. Default to false.
            - api_version (str | None): The api version. Default to None. E.g., "2024-02-01".

        Example:
            >>> # filter_func example:
            >>> def yes_or_no_filter(context, response):
            >>>    return context.get("yes_or_no_choice", False) is False or any(
            >>>        text in ["Yes.", "No."] for text in client.extract_text_or_completion_object(response)
            >>>    )

        Raises:
            - RuntimeError: If all declared custom model clients are not registered
            - APIError: If any model client create call raises an APIError
        """
        return self._model.create(**config)


def auto_construct_oai_config_list_from_env() -> List:
    """
    Collect various API keys saved in the environment and return a format like:
    [{"model": "gpt-4", "api_key": xxx}, {"model": "claude-3.5-sonnet", "api_key": xxx}]

    Note this is a lazy function that defaults to gpt-40 and claude-3.5-sonnet.
    If you want to specify your own model, please provide an OAI_CONFIG_LIST in the environment or as a file
    """
    config_list = []
    if os.environ.get("OPENAI_API_KEY") is not None:
        config_list.append(
            {"model": "gpt-4o", "api_key": os.environ.get("OPENAI_API_KEY")}
        )
    if os.environ.get("ANTHROPIC_API_KEY") is not None:
        config_list.append(
            {
                "model": "claude-3-5-sonnet-latest",
                "api_key": os.environ.get("ANTHROPIC_API_KEY"),
            }
        )
    return config_list


class LiteLLM(AbstractModel):
    """
    This is an LLM backend supported by LiteLLM library.

    https://docs.litellm.ai/docs/completion/input

    To use this, set the credentials through the environment variable as
    instructed in the LiteLLM documentation. For convenience, you can set the
    default model name through the environment variable TRACE_LITELLM_MODEL.
    When using Azure models via token provider, you can set the Azure token
    provider scope through the environment variable AZURE_TOKEN_PROVIDER_SCOPE.
    
    This class now supports storing default completion parameters (like temperature,
    top_p, max_tokens, etc.) that will be used for all calls unless overridden.
    """

    def __init__(self, model: Union[str, None] = None, reset_freq: Union[int, None] = None,
                 cache=True, max_retries=10, base_delay=1.0, **default_params) -> None:
        if model is None:
            model = os.environ.get('TRACE_LITELLM_MODEL')
            if model is None:
                # warnings.warn("TRACE_LITELLM_MODEL environment variable is not found when loading the default model for LiteLLM. Attempt to load the default model from DEFAULT_LITELLM_MODEL environment variable. The usage of DEFAULT_LITELLM_MODEL will be deprecated. Please use the environment variable TRACE_LITELLM_MODEL for setting the default model name for LiteLLM.")
                model = os.environ.get('DEFAULT_LITELLM_MODEL', 'gpt-4o')

        self.model_name = model
        self.cache = cache
        self.default_params = default_params  # Store default completion parameters
        factory = lambda: self._factory(self.model_name, self.default_params, max_retries=max_retries, base_delay=base_delay)
        super().__init__(factory, reset_freq)

    @classmethod
    def _factory(cls, model_name: str, default_params: dict, max_retries=10, base_delay=1.0):
        import litellm
        if model_name.startswith('azure/'):  # azure model
            azure_token_provider_scope = os.environ.get('AZURE_TOKEN_PROVIDER_SCOPE', None)
            if azure_token_provider_scope is not None:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider
                credential = get_bearer_token_provider(DefaultAzureCredential(), azure_token_provider_scope)
                return lambda *args, **kwargs: retry_with_exponential_backoff(
                    lambda: litellm.completion(model_name, *args,
                                             azure_ad_token_provider=credential, **{**default_params, **kwargs}),
                    max_retries=max_retries,
                    base_delay=base_delay,
                    operation_name="LiteLLM_completion"
                )
        return lambda *args, **kwargs: retry_with_exponential_backoff(
            lambda: litellm.completion(model_name, *args, **{**default_params, **kwargs}),
            max_retries=max_retries,
            base_delay=base_delay,
            operation_name="LiteLLM_completion"
        )

    @property
    def model(self):
        """
        response = litellm.completion(
            model=self.model,
            messages=[{"content": message, "role": "user"}]
        )
        """
        return lambda *args, **kwargs: self._model(*args, **kwargs)


class CustomLLM(AbstractModel):
    """
    This is for Custom server's API endpoints that are OpenAI Compatible.
    Such server includes LiteLLM proxy server.
    """

    def __init__(self, model: Union[str, None] = None, reset_freq: Union[int, None] = None,
                 cache=True) -> None:
        if model is None:
            model = os.environ.get('TRACE_CUSTOMLLM_MODEL', 'gpt-4o')
        base_url = os.environ.get('TRACE_CUSTOMLLM_URL', 'http://xx.xx.xxx.xx:4000/')
        server_api_key = os.environ.get('TRACE_CUSTOMLLM_API_KEY',
                                        'sk-Xhg...')  # we assume the server has an API key
        # the server API is set through `master_key` in `config.yaml` for LiteLLM proxy server

        self.model_name = model
        self.cache = cache
        factory = lambda: self._factory(base_url, server_api_key)  # an LLM instance uses a fixed model
        super().__init__(factory, reset_freq)

    @classmethod
    def _factory(cls, base_url: str, server_api_key: str):
        import openai
        return openai.OpenAI(base_url=base_url, api_key=server_api_key)

    @property
    def model(self):
        return lambda **kwargs: self.create(**kwargs)
        # return lambda *args, **kwargs: self._model.chat.completions.create(*args, **kwargs)

    def create(self, **config: Any):
        if 'model' not in config:
            config['model'] = self.model_name
        return self._model.chat.completions.create(**config)

# Registry of available backends
_LLM_REGISTRY = {
    "LiteLLM": LiteLLM,
    "AutoGen": AutoGenLLM,
    "CustomLLM": CustomLLM,
}

class LLMFactory:
    """Factory for creating LLM instances with named profiles.

    Profiles allow you to save and reuse LLM configurations with specific settings.
    Each profile can include any LiteLLM-supported parameters like model, temperature,
    top_p, max_tokens, etc.

    The default profile uses 'gpt-4o-mini' with standard settings.

    Basic Usage:
        # Use default model (gpt-4o-mini)
        llm = LLM()
        
        # Specify a model directly
        llm = LLM(model="gpt-4o")
        
        # Use a named profile
        llm = LLM(profile="my_profile")

    Creating Custom Profiles:
        # Register a profile with full LiteLLM configuration
        LLMFactory.create_profile(
            "creative_writer",
            backend="LiteLLM",
            model="gpt-4o",
            temperature=0.9,
            top_p=0.95,
            max_tokens=2000,
            presence_penalty=0.6
        )
        
        # Register a reasoning profile
        LLMFactory.create_profile(
            "deep_thinker",
            backend="LiteLLM",
            model="o1-preview",
            max_completion_tokens=8000
        )
        
        # Register a profile with specific formatting
        LLMFactory.create_profile(
            "json_responder",
            backend="LiteLLM",
            model="gpt-4o-mini",
            temperature=0.3,
            response_format={"type": "json_object"}
        )

    Using Profiles:
        # Use your custom profile
        llm = LLM(profile="creative_writer")
        
        # In optimizers
        optimizer = OptoPrime(parameters, llm=LLM(profile="deep_thinker"))

    Profile Management:
        # List all available profiles
        profiles = LLMFactory.list_profiles()
        
        # Get profile configuration
        config = LLMFactory.get_profile_info("creative_writer")
        
        # Override existing profile
        LLMFactory.create_profile("default", "LiteLLM", model="gpt-4o", temperature=0.5)

    Supported LiteLLM Parameters:
        See https://docs.litellm.ai/docs/completion/input for full list:
        - model: Model name (required)
        - temperature: Sampling temperature (0-2)
        - top_p: Nucleus sampling parameter
        - max_tokens: Maximum tokens to generate
        - max_completion_tokens: Upper bound for completion tokens
        - presence_penalty: Penalize new tokens based on presence
        - frequency_penalty: Penalize new tokens based on frequency
        - stop: Stop sequences (string or list)
        - stream: Enable streaming responses
        - response_format: Output format specification
        - seed: Deterministic sampling seed
        - tools: Function calling tools
        - tool_choice: Control function calling behavior
        - logprobs: Return log probabilities
        - top_logprobs: Number of most likely tokens to return
        - n: Number of completions to generate
        - and many more...
    """

    # Default profile - just gpt-4o-mini with no opinionated settings
    _profiles = {
        'default': {'backend': 'LiteLLM', 'params': {'model': 'gpt-4o-mini'}},
    }

    @classmethod
    def get_llm(cls, profile: str = 'default', model: str = None, **kwargs) -> AbstractModel:
        """Get an LLM instance for the specified profile or model.
        
        Args:
            profile: Name of the profile to use. Defaults to 'default'.
            model: Model name to use directly. If provided, overrides profile.
            **kwargs: Additional parameters to pass to the backend (e.g., temperature, top_p).
                     These override profile settings if both are specified.
        
        Returns:
            An LLM instance configured according to the profile/model and parameters.
        
        Examples:
            # Use default profile
            llm = LLMFactory.get_llm()
            
            # Use specific model
            llm = LLMFactory.get_llm(model="gpt-4o")
            
            # Use named profile
            llm = LLMFactory.get_llm(profile="creative_writer")
            
            # Use model with custom parameters
            llm = LLMFactory.get_llm(model="gpt-4o", temperature=0.7, max_tokens=1000)
            
            # Override profile settings
            llm = LLMFactory.get_llm(profile="creative_writer", temperature=0.5)
        """
        # If model is specified directly, create a simple config
        if model is not None:
            backend = kwargs.pop('backend', None)
            if backend is None:
                backend = 'LiteLLM'
            backend_cls = _LLM_REGISTRY[backend]
            params = {'model': model, **kwargs}
            return backend_cls(**params)
        
        # Otherwise use profile
        if profile not in cls._profiles:
            raise ValueError(
                f"Unknown profile '{profile}'. Available profiles: {list(cls._profiles.keys())}. "
                f"Use LLMFactory.create_profile() to create custom profiles, or pass model= directly."
            )

        config = cls._profiles[profile].copy()
        backend_cls = _LLM_REGISTRY[config['backend']]
        
        # Merge profile params with any override kwargs
        params = config['params'].copy()
        params.update(kwargs)
        
        return backend_cls(**params)

    @classmethod
    def create_profile(cls, name: str, backend: str = 'LiteLLM', **params):
        """Register a new LLM profile with custom configuration.
        
        Args:
            name: Profile name to register.
            backend: Backend to use ('LiteLLM', 'AutoGen', or 'CustomLLM'). Defaults to 'LiteLLM'.
            **params: Configuration parameters for the backend. For LiteLLM, this can include
                     any parameters from https://docs.litellm.ai/docs/completion/input
        
        Examples:
            # Simple profile with just a model
            LLMFactory.create_profile("gpt4", model="gpt-4o")
            
            # Profile with temperature and token settings
            LLMFactory.create_profile(
                "creative",
                model="gpt-4o",
                temperature=0.9,
                max_tokens=2000
            )
            
            # Profile with advanced settings
            LLMFactory.create_profile(
                "structured_json",
                model="gpt-4o-mini",
                temperature=0.3,
                response_format={"type": "json_object"},
                max_tokens=1500,
                top_p=0.9
            )
        """
        if backend not in _LLM_REGISTRY:
            raise ValueError(
                f"Unknown backend '{backend}'. Valid options: {list(_LLM_REGISTRY.keys())}"
            )
        cls._profiles[name] = {'backend': backend, 'params': params}

    @classmethod
    def list_profiles(cls):
        """List all available profile names."""
        return list(cls._profiles.keys())

    @classmethod
    def get_profile_info(cls, profile: str = None):
        """Get configuration information about one or all profiles.
        
        Args:
            profile: Profile name to get info for. If None, returns all profiles.
        
        Returns:
            Dictionary with profile configuration(s).
        """
        if profile:
            return cls._profiles.get(profile)
        return cls._profiles


class DummyLLM(AbstractModel):
    """A dummy LLM that does nothing. Used for testing purposes."""

    def __init__(self,
                 callable,
                 reset_freq: Union[int, None] = None) -> None:
        # self.message = message
        self.callable = callable
        super().__init__(self._factory, reset_freq)

    def _factory(self):

        # set response.choices[0].message.content
        # create a fake container with above format

        class Message:
            def __init__(self, content):
                self.content = content
        class Choice:
            def __init__(self, content):
                self.message = Message(content)
        class Response:
            def __init__(self, content):
                self.choices = [Choice(content)]

        return lambda *args, **kwargs:  Response(self.callable(*args, **kwargs))


class LLM:
    """
    A unified entry point for all supported LLM backends.

    The LLM class provides a simple interface for creating language model instances.
    By default, it uses gpt-4o-mini through LiteLLM.

    Basic Usage:
        # Use default model (gpt-4o-mini)
        llm = LLM()
        
        # Specify a model directly
        llm = LLM(model="gpt-4o")
        llm = LLM(model="claude-3-5-sonnet-latest")
        llm = LLM(model="o1-preview")
        
        # Add LiteLLM parameters
        llm = LLM(model="gpt-4o", temperature=0.7, max_tokens=2000)
        llm = LLM(model="gpt-4o-mini", temperature=0.3, top_p=0.9)

    Using Named Profiles:
        # Use a saved profile
        llm = LLM(profile="my_custom_profile")
        
        # Create profiles with LLMFactory
        LLMFactory.create_profile("creative", model="gpt-4o", temperature=0.9)
        llm = LLM(profile="creative")

    Using Different Backends:
        # Explicitly specify backend (default: LiteLLM)
        llm = LLM(backend="AutoGen", config_list=my_configs)
        llm = LLM(backend="CustomLLM", model="llama-3.1-8b")
        
        # Or set via environment variable
        # export TRACE_DEFAULT_LLM_BACKEND=AutoGen
        llm = LLM()

    Examples with LiteLLM Parameters:
        # Structured output
        llm = LLM(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        # High creativity
        llm = LLM(
            model="gpt-4o",
            temperature=0.9,
            top_p=0.95,
            presence_penalty=0.6
        )
        
        # Deterministic responses
        llm = LLM(
            model="gpt-4o-mini",
            temperature=0,
            seed=42
        )

    See Also:
        - LLMFactory: For managing named profiles
        - https://docs.litellm.ai/docs/completion/input: Full list of LiteLLM parameters
    """
    def __new__(cls, model: str = None, profile: str = None, backend: str = None, **kwargs):
        # Priority 1: If profile is specified, use LLMFactory
        if profile:
            return LLMFactory.get_llm(profile=profile, **kwargs)
        
        # Priority 2: If model is specified, use LLMFactory with model
        if model:
            if backend is not None:
                kwargs['backend'] = backend
            return LLMFactory.get_llm(model=model, **kwargs)
        
        # Priority 3: Use backend-specific instantiation (for AutoGen, CustomLLM, etc.)
        # This path is for when neither profile nor model is specified
        name = backend or os.getenv("TRACE_DEFAULT_LLM_BACKEND", "LiteLLM")
        try:
            backend_cls = _LLM_REGISTRY[name]
        except KeyError:
            raise ValueError(f"Unknown LLM backend: {name}. "
                             f"Valid options are: {list(_LLM_REGISTRY)}")
        # Instantiate and return the chosen subclass
        return backend_cls(**kwargs)