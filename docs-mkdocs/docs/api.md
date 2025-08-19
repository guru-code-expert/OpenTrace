# 📖 API Reference

This section contains the complete API documentation for OpenTrace, automatically generated from the source code docstrings.

## Quick Navigation

- **[opto.trace](api/trace/index.md)** - Core tracing functionality, nodes, and graph operations
- **[opto.optimizers](api/optimizers/index.md)** - Optimization algorithms and utilities
- **[opto.trainer](api/trainer/index.md)** - Training workflows and evaluation tools
- **[opto.utils](api/utils/index.md)** - Utility functions and helpers
- **[opto.features](api/features/index.md)** - Additional features like MLflow integration

## Getting Started with the API

The main entry point is the `opto.trace` module:

```python
import opto.trace as trace

# Create a node
node = trace.node("Hello, World!")

# Use with optimizers
from opto.optimizers import OptimizerManager
optimizer = OptimizerManager()
```

Browse the full API documentation using the navigation above or search for specific functions and classes.