# Introducing Trainer

OpenTrace v0.2 introduces the powerful **Trainer** module, a comprehensive framework for training and optimizing AI agents with advanced algorithms and evaluation capabilities.

## 🎯 Overview

The Trainer module provides a unified interface for:
- **Algorithm Selection**: Choose from various optimization algorithms
- **Evaluation Metrics**: Comprehensive performance assessment
- **Learning Strategies**: Advanced training methodologies
- **Data Loading**: Efficient batch processing and data management

## 🚀 Key Features

### Advanced Algorithms
The Trainer supports multiple state-of-the-art optimization algorithms:

```python
from opto.trainer import Trainer
from opto.trainer.algorithms import BeamSearchAlgorithm, UCBSearch

# Initialize trainer with beam search
trainer = Trainer(
    algorithm=BeamSearchAlgorithm(beam_size=10),
    evaluator=your_evaluator,
    max_iterations=100
)

# Or use UCB for exploration-exploitation balance
trainer = Trainer(
    algorithm=UCBSearch(confidence=0.95),
    evaluator=your_evaluator
)
```

### Comprehensive Evaluation
Built-in evaluators provide detailed performance metrics:

```python
from opto.trainer.evaluators import MultiMetricEvaluator

evaluator = MultiMetricEvaluator([
    'accuracy',
    'latency', 
    'cost',
    'robustness'
])

trainer = Trainer(
    algorithm=your_algorithm,
    evaluator=evaluator
)
```

### Data Loading and Preprocessing
Efficient data handling for training workflows:

```python
from opto.trainer.loader import DataLoader

loader = DataLoader(
    batch_size=32,
    shuffle=True,
    preprocessing_fn=your_preprocessing
)

trainer.fit(loader)
```

## 📊 Training Workflows

### Basic Training Loop
```python
# Simple training setup
trainer = Trainer(
    algorithm=BeamSearchAlgorithm(),
    evaluator=your_evaluator,
    logger=ConsoleLogger()
)

# Train your agent
results = trainer.fit(
    train_data=train_loader,
    validation_data=val_loader,
    epochs=50
)

print(f"Best performance: {results.best_score}")
```

### Advanced Configuration
```python
# Advanced trainer with custom settings
trainer = Trainer(
    algorithm=UCBSearch(
        confidence=0.95,
        exploration_weight=0.1
    ),
    evaluator=MultiMetricEvaluator(['accuracy', 'efficiency']),
    logger=MLFlowLogger(),
    early_stopping=True,
    patience=10
)

# Custom training callbacks
trainer.add_callback('on_epoch_end', custom_callback)
trainer.fit(data_loader)
```

## 🔧 Customization

### Custom Algorithms
Extend the framework with your own algorithms:

```python
from opto.trainer.algorithms import Algorithm

class CustomAlgorithm(Algorithm):
    def __init__(self, custom_param=1.0):
        self.custom_param = custom_param
    
    def suggest(self, history):
        # Your custom suggestion logic
        return suggested_parameters
    
    def update(self, parameters, score):
        # Update algorithm state
        pass
```

### Custom Evaluators
Create domain-specific evaluation metrics:

```python
from opto.trainer.evaluators import Evaluator

class DomainSpecificEvaluator(Evaluator):
    def evaluate(self, agent_output, ground_truth):
        # Your custom evaluation logic
        return {
            'custom_metric': score,
            'additional_info': metadata
        }
```

## 📈 Integration with Existing Code

The Trainer seamlessly integrates with your existing OpenTrace workflows:

```python
import opto

@opto.trace
def my_agent(query):
    # Your existing agent code
    return response

# Wrap with trainer for optimization
trainer = Trainer(
    target_function=my_agent,
    algorithm=BeamSearchAlgorithm(),
    evaluator=your_evaluator
)

# Optimize your agent
optimized_agent = trainer.optimize()
```

## 🎯 Best Practices

1. **Start Simple**: Begin with basic algorithms and gradually add complexity
2. **Monitor Training**: Use comprehensive logging and visualization tools  
3. **Validate Frequently**: Implement robust validation strategies
4. **Experiment**: Try different algorithms and hyperparameters
5. **Scale Gradually**: Start with small datasets and scale up

## 📚 Learn More

- [Trainer API Reference](../api/trainer/index.md)
- [Training Tutorial](../tutorials/trainers.ipynb)
- [Algorithm Comparison Guide](../api/trainer/algorithms/index.md)
- [Custom Evaluators Examples](../api/trainer/evaluators.md)

Ready to start training? Check out our [comprehensive trainer tutorial](../tutorials/trainers.ipynb) for hands-on examples!