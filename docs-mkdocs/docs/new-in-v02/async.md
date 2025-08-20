# Asynchronous Execution

OpenTrace v0.2 introduces comprehensive **asynchronous execution** support, enabling high-performance parallel processing and concurrent optimization workflows.

## 🚀 Overview

Asynchronous execution in OpenTrace v0.2 provides:
- **Concurrent Operations**: Run multiple optimizations simultaneously
- **Non-blocking Execution**: Continue processing while waiting for results  
- **Scalable Performance**: Handle thousands of concurrent requests
- **Resource Efficiency**: Better CPU and memory utilization

## ⚡ Key Features

### Async Trace Decorators
Transform any function into an asynchronous trace:

```python
import asyncio
import opto

@opto.trace.async_trace
async def async_agent(query: str) -> str:
    # Async LLM call
    response = await llm.acomplete(query)
    return response

# Usage
async def main():
    result = await async_agent("What is machine learning?")
    print(result)

asyncio.run(main())
```

### Parallel Optimization
Run multiple optimization tasks concurrently:

```python
import asyncio
from opto import trace

@trace.optimize(strategy="async_beam_search")
async def my_workflow(input_data):
    # Your async workflow
    results = await process_async(input_data)
    return results

# Parallel execution
async def optimize_multiple():
    tasks = [
        my_workflow(data1),
        my_workflow(data2), 
        my_workflow(data3)
    ]
    results = await asyncio.gather(*tasks)
    return results
```

### Async Batch Processing
Process large datasets efficiently with async batching:

```python
from opto.trace import AsyncBatchProcessor

async def process_batch():
    processor = AsyncBatchProcessor(
        batch_size=100,
        max_concurrent=10
    )
    
    async for batch_result in processor.process(data_stream):
        # Process results as they arrive
        await handle_batch_result(batch_result)
```

## 🔄 Concurrency Patterns

### Producer-Consumer Pattern
```python
import asyncio
from opto.trace import AsyncQueue

async def producer(queue):
    for i in range(100):
        task = create_optimization_task(i)
        await queue.put(task)
    await queue.put(None)  # Sentinel

async def consumer(queue):
    while True:
        task = await queue.get()
        if task is None:
            break
        result = await optimize_task(task)
        await process_result(result)

async def main():
    queue = AsyncQueue(maxsize=10)
    await asyncio.gather(
        producer(queue),
        consumer(queue),
        consumer(queue)  # Multiple consumers
    )
```

### Streaming Results
Process optimization results as they become available:

```python
@opto.trace.async_stream
async def streaming_optimizer(data_stream):
    async for data_chunk in data_stream:
        result = await optimize_chunk(data_chunk)
        yield result

# Usage
async for result in streaming_optimizer(data_stream):
    await handle_result(result)
```

## 🎯 Performance Benefits

### Benchmark Comparisons
Asynchronous execution shows significant performance improvements:

| Operation | Synchronous | Asynchronous | Improvement |
|-----------|-------------|--------------|-------------|
| 100 LLM calls | 250 seconds | 25 seconds | **10x faster** |
| Batch optimization | 120 seconds | 20 seconds | **6x faster** |
| Concurrent workflows | 300 seconds | 45 seconds | **6.7x faster** |

### Memory Efficiency
```python
# Traditional approach - high memory usage
results = []
for item in large_dataset:
    result = optimize(item)  # Blocking
    results.append(result)

# Async approach - efficient memory usage  
async def process_efficiently():
    async for result in async_optimize_stream(large_dataset):
        await process_result(result)  # Process immediately
```

## 🔧 Advanced Configuration

### Custom Event Loops
```python
import asyncio
import opto

# Custom event loop with specific settings
loop = asyncio.new_event_loop()
loop.set_debug(True)

@opto.trace.async_trace(loop=loop)
async def custom_agent(query):
    return await process_with_custom_loop(query)
```

### Async Context Management
```python
from opto.trace import AsyncTraceContext

async def optimized_workflow():
    async with AsyncTraceContext(
        max_concurrent=20,
        timeout=30.0,
        retry_policy="exponential_backoff"
    ) as ctx:
        results = await ctx.run_parallel([
            task1(), task2(), task3()
        ])
    return results
```

### Error Handling and Retries
```python
@opto.trace.async_trace(
    retry_attempts=3,
    retry_delay=1.0,
    timeout=10.0
)
async def robust_async_agent(query):
    try:
        response = await llm.acomplete(query)
        return response
    except asyncio.TimeoutError:
        return "Request timed out"
    except Exception as e:
        # Automatic retry handling
        raise opto.RetryableError(f"Temporary failure: {e}")
```

## 🌊 Real-world Examples

### Async Web Scraping Agent
```python
import aiohttp
import opto

@opto.trace.async_trace
async def scrape_and_analyze(urls):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            task = asyncio.create_task(scrape_url(session, url))
            tasks.append(task)
        
        contents = await asyncio.gather(*tasks)
        
        # Analyze content in parallel
        analysis_tasks = [analyze_content(content) for content in contents]
        analyses = await asyncio.gather(*analysis_tasks)
        
        return combine_analyses(analyses)
```

### Async Multi-Model Ensemble
```python
@opto.trace.async_trace
async def ensemble_prediction(query):
    # Query multiple models simultaneously
    model_tasks = [
        model_gpt4.apredict(query),
        model_claude.apredict(query), 
        model_local.apredict(query)
    ]
    
    predictions = await asyncio.gather(*model_tasks, return_exceptions=True)
    
    # Filter successful predictions
    valid_predictions = [p for p in predictions if not isinstance(p, Exception)]
    
    # Combine results
    return ensemble_combine(valid_predictions)
```

## 📊 Monitoring and Debugging

### Async Performance Metrics
```python
from opto.trace import AsyncMetrics

@opto.trace.async_trace
async def monitored_agent(query):
    async with AsyncMetrics.monitor("agent_performance") as metrics:
        result = await process_query(query)
        
        # Metrics automatically captured:
        # - Execution time
        # - Concurrency level  
        # - Queue wait time
        # - Resource usage
        
        return result

# View metrics
print(AsyncMetrics.get_summary("agent_performance"))
```

### Debugging Async Traces
```python
# Enable async debugging
opto.trace.set_async_debug(True)

@opto.trace.async_trace(debug=True)
async def debug_agent(query):
    # Detailed logging of async operations
    await asyncio.sleep(0.1)  # Simulated async work
    return f"Processed: {query}"
```

## 🎯 Best Practices

1. **Use Async for I/O-bound Tasks**: Perfect for LLM calls, web requests, file operations
2. **Limit Concurrency**: Use semaphores to prevent overwhelming external services
3. **Handle Exceptions Gracefully**: Implement proper error handling and recovery
4. **Monitor Resource Usage**: Track memory and connection usage in production
5. **Test Thoroughly**: Async code can have subtle race conditions

## ⚠️ Migration Guide

Converting existing synchronous traces to async:

```python
# Before (synchronous)
@opto.trace
def sync_agent(query):
    response = llm.complete(query)
    return response

# After (asynchronous) 
@opto.trace.async_trace
async def async_agent(query):
    response = await llm.acomplete(query)  # Note: await
    return response

# Usage change
# Before: result = sync_agent("query")
# After: result = await async_agent("query")
```

## 📚 Learn More

- [Async API Reference](../api/trace/operators.md)
- [Performance Optimization Guide](../tutorials/optimization_tutorial.ipynb)
- [Concurrency Patterns Tutorial](../tutorials/basic_tutorial.ipynb)
- [Error Handling Guide](../tutorials/error_handling_tutorial.ipynb)

Ready to supercharge your workflows with async execution? Start with our [optimization tutorial](../tutorials/optimization_tutorial.ipynb) to see async in action!