# Installation

Get started with OpenTrace in just a few steps. Choose the installation method that works best for your needs.

## Quick Installation

!!! tip "Recommended"
    Install OpenTrace from PyPI for the latest stable release:
    ```bash
    pip install trace-opt
    ```

## Development Installation

!!! info "For Contributors"
    Clone the repository and install in editable mode to contribute or access the latest features:
    ```bash
    git clone https://github.com/AgentOpt/OpenTrace.git
    cd OpenTrace
    pip install -e .
    ```

## Requirements

!!! warning "System Requirements"
    - **Python 3.8+** - OpenTrace requires Python 3.8 or higher
    - **No core dependencies** - Basic tracing functionality works out of the box
    - **Optional: LiteLLM** - Required only if using `opto.optimizers` for LLM API calls

## Package Components

OpenTrace is designed with a modular architecture:

### Core Tracing (`opto.trace`)
- **Zero dependencies** - Pure Python implementation
- **Execution graph capture** - Records your workflow automatically
- **Works offline** - No external API calls required

### Optimizers (`opto.optimizers`)
- **LLM-powered optimization** - Requires LiteLLM for API calls
- **Multiple LLM providers** - Support for OpenAI, Anthropic, and more
- **Generative improvements** - Automatically enhance your workflows

## Verify Installation

Test your installation with a quick example:

```python
from opto import trace

@trace.model
class SimpleAgent:
    def __init__(self):
        self.instruction = trace.node("Be helpful", trainable=True)
    
    def greet(self, name):
        return f"Hello, {name}!"

# Create and test the agent
agent = SimpleAgent()
result = agent.greet("World")
print(result)  # Output: Hello, World!
```

## Next Steps

- **New to OpenTrace?** Check out our [Quick Start Guide](quick_start.ipynb)
- **Want to see examples?** Explore our [Examples](../examples/basic/greeting.ipynb) section
- **Need help?** Visit our [FAQ](../faq/faq.md) or [API Reference](../api.md)

[:material-rocket-launch: Start with Quick Start](quick_start.ipynb){ .md-button .md-button--primary }
[:material-book-open: View Examples](../examples/basic/greeting.ipynb){ .md-button }