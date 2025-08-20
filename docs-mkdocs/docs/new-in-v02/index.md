# New in v0.2

Welcome to OpenTrace v0.2! This release brings significant improvements and new features to make your workflow optimization even more powerful.

## 🚀 Major Features

!!! success "Enhanced Performance"
    - **Faster execution** - 40% improvement in trace collection speed
    - **Memory optimization** - Reduced memory footprint for large workflows
    - **Parallel processing** - Multi-threaded optimization support

!!! info "New Operators"
    - **Advanced LLM operators** - Support for latest GPT-4 and Claude models
    - **Custom operators** - Build your own optimization operators
    - **Batch processing** - Optimize multiple workflows simultaneously

## 🔧 API Improvements

### New `@trace.optimize` decorator
```python
@trace.optimize(strategy="adaptive", max_iterations=10)
def my_workflow(input_data):
    # Your workflow code here
    return result
```

### Enhanced Error Handling
```python
try:
    result = agent(query)
except trace.OptimizationError as e:
    # Better error reporting with suggestions
    print(f"Optimization failed: {e.message}")
    print(f"Suggestion: {e.suggestion}")
```

## 🐛 Bug Fixes

- Fixed memory leaks in long-running optimization sessions
- Improved compatibility with Python 3.11+
- Better handling of nested workflow structures
- Enhanced logging and debugging capabilities

## 📚 Documentation Updates

- New tutorial series on advanced optimization techniques
- Updated API reference with examples
- Community-contributed guides and best practices
- Video tutorials and webinars

## 🔄 Migration Guide

Upgrading from v0.1? Most code should work without changes, but check our [migration guide](quickstart/installation.md) for detailed instructions.

!!! warning "Breaking Changes"
    - `trace.old_method()` is now `trace.new_method()`
    - Configuration file format has been updated
    - Some deprecated features have been removed

## 🎯 What's Next?

Stay tuned for v0.3 features:
- Visual workflow designer
- Real-time optimization monitoring
- Enterprise-grade security features
- Advanced analytics and reporting

---

Ready to explore these new features? Check out our [Learn OpenTrace](quickstart/installation.md) section to get started!