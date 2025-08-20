# MLFlow Integration

OpenTrace v0.2 introduces seamless **MLFlow integration**, enabling comprehensive experiment tracking, model management, and deployment workflows for your AI optimization projects.

## 🎯 Overview

MLFlow integration provides:
- **Experiment Tracking**: Automatic logging of optimization runs and metrics
- **Model Registry**: Version control and management for optimized agents
- **Deployment**: Easy deployment of optimized models to production
- **Visualization**: Rich dashboards and comparative analysis

## 🚀 Key Features

### Automatic Experiment Logging
OpenTrace automatically logs optimization experiments to MLFlow:

```python
import opto
import mlflow

# Enable MLFlow integration
opto.mlflow.enable_autolog()

@opto.trace.optimize(strategy="beam_search", max_iterations=10)
def optimized_agent(query: str) -> str:
    # Your agent implementation
    response = llm.complete(query)
    return response

# Optimization automatically logged to MLFlow
result = optimized_agent.optimize(train_data)

# View in MLFlow UI: mlflow ui
```

### Custom Metrics Tracking
Log custom metrics and parameters:

```python
import opto.mlflow as opto_mlflow

@opto.trace
def my_agent(query: str) -> str:
    with opto_mlflow.start_run():
        # Log parameters
        opto_mlflow.log_param("model", "gpt-4")
        opto_mlflow.log_param("temperature", 0.7)
        
        response = llm.complete(query, temperature=0.7)
        
        # Log metrics
        opto_mlflow.log_metric("response_length", len(response))
        opto_mlflow.log_metric("cost", calculate_cost(response))
        
        return response
```

### Model Registry Integration
Register and manage optimized models:

```python
from opto.mlflow import ModelRegistry

# Train and optimize your agent
trainer = opto.Trainer(
    algorithm=BeamSearchAlgorithm(),
    evaluator=your_evaluator
)
optimized_agent = trainer.fit(training_data)

# Register the optimized model
registry = ModelRegistry()
model_version = registry.register_model(
    name="optimized-qa-agent",
    model=optimized_agent,
    description="Optimized Q&A agent with beam search",
    tags={"version": "v0.2", "algorithm": "beam_search"}
)

print(f"Model registered as version {model_version}")
```

## 📊 Experiment Tracking

### Automatic Logging
OpenTrace automatically tracks key optimization metrics:

```python
# This code automatically logs to MLFlow:
@opto.trace.optimize(
    strategy="adaptive",
    max_iterations=50,
    mlflow_experiment="agent-optimization"
)
def research_agent(query: str) -> str:
    # Automatically logged:
    # - Iteration number
    # - Parameter values
    # - Performance scores
    # - Execution time
    # - Resource usage
    
    return process_research_query(query)
```

### Custom Experiment Configuration
```python
import mlflow
import opto

# Set up custom MLFlow experiment
mlflow.set_experiment("advanced-agent-optimization")

with opto.mlflow.experiment_context("custom-run-name"):
    # Configure logging
    opto.mlflow.configure({
        "log_model_checkpoints": True,
        "log_optimization_trace": True,
        "log_performance_metrics": True,
        "artifact_location": "s3://my-bucket/experiments"
    })
    
    # Run optimization
    result = optimize_agent(training_data)
```

### Comparative Analysis
```python
# Compare multiple optimization runs
experiments = [
    {"name": "beam_search", "strategy": "beam_search", "beam_size": 5},
    {"name": "ucb_search", "strategy": "ucb", "confidence": 0.95},
    {"name": "random_search", "strategy": "random", "samples": 100}
]

results = {}
for exp in experiments:
    with opto.mlflow.start_run(run_name=exp["name"]):
        # Log experiment parameters
        opto.mlflow.log_params(exp)
        
        # Run optimization
        result = optimize_agent(strategy=exp["strategy"])
        results[exp["name"]] = result
        
        # Log results
        opto.mlflow.log_metric("final_score", result.best_score)
        opto.mlflow.log_metric("iterations", result.iterations)

# View comparison in MLFlow UI
```

## 🎛️ Advanced Configuration

### Custom MLFlow Backend
```python
# Configure remote MLFlow server
opto.mlflow.set_tracking_uri("https://your-mlflow-server.com")
opto.mlflow.set_registry_uri("s3://your-model-registry")

# Use custom artifact store
opto.mlflow.set_artifact_location("gs://your-gcs-bucket/artifacts")
```

### Integration with Cloud Providers

#### AWS Integration
```python
import opto.mlflow.aws as aws_mlflow

# Use AWS-specific configuration
aws_mlflow.configure_s3_backend(
    bucket="your-s3-bucket",
    region="us-west-2",
    access_key="your-access-key",
    secret_key="your-secret-key"
)

# Automatic logging to S3
@opto.trace.optimize(mlflow_backend="aws")
def aws_optimized_agent(query):
    return your_agent_logic(query)
```

#### Google Cloud Integration
```python
import opto.mlflow.gcp as gcp_mlflow

# Configure GCP backend
gcp_mlflow.configure_gcs_backend(
    bucket="your-gcs-bucket",
    project_id="your-project-id",
    credentials_path="path/to/service-account.json"
)
```

#### Azure Integration
```python
import opto.mlflow.azure as azure_mlflow

# Configure Azure ML backend
azure_mlflow.configure_azure_backend(
    subscription_id="your-subscription-id",
    resource_group="your-resource-group",
    workspace_name="your-workspace"
)
```

## 🚀 Model Deployment

### Local Deployment
```python
from opto.mlflow import ModelDeployment

# Load registered model
deployment = ModelDeployment.from_registry(
    model_name="optimized-qa-agent",
    version="latest"
)

# Deploy locally
server = deployment.deploy_local(port=8000)
print("Model serving at http://localhost:8000")

# Test deployment
response = deployment.predict({"query": "What is machine learning?"})
```

### Production Deployment
```python
# Deploy to cloud platform
cloud_deployment = deployment.deploy_cloud(
    platform="aws-sagemaker",  # or "gcp-vertex", "azure-ml"
    instance_type="ml.m5.large",
    min_instances=1,
    max_instances=10
)

print(f"Model deployed at: {cloud_deployment.endpoint_url}")
```

### Batch Prediction
```python
# Run batch predictions
batch_job = deployment.create_batch_job(
    input_data="s3://your-bucket/input-data.json",
    output_location="s3://your-bucket/predictions/",
    instance_count=5
)

# Monitor job progress
while not batch_job.is_complete():
    print(f"Job status: {batch_job.status}")
    time.sleep(30)

print(f"Predictions saved to: {batch_job.output_location}")
```

## 📈 Monitoring and Alerting

### Performance Monitoring
```python
from opto.mlflow import ModelMonitor

# Set up model monitoring
monitor = ModelMonitor(
    model_name="optimized-qa-agent",
    version="production"
)

# Configure alerts
monitor.add_alert(
    metric="accuracy",
    threshold=0.85,
    condition="less_than",
    action="email",
    recipients=["team@company.com"]
)

monitor.add_alert(
    metric="response_time",
    threshold=2000,  # milliseconds
    condition="greater_than",
    action="slack",
    webhook_url="https://hooks.slack.com/your-webhook"
)

# Start monitoring
monitor.start()
```

### A/B Testing
```python
from opto.mlflow import ABTest

# Set up A/B test between model versions
ab_test = ABTest(
    name="agent-optimization-test",
    model_a={"name": "optimized-qa-agent", "version": "v1.0"},
    model_b={"name": "optimized-qa-agent", "version": "v2.0"},
    traffic_split=0.5,  # 50/50 split
    success_metric="user_satisfaction"
)

# Deploy A/B test
ab_test.deploy()

# Monitor results
results = ab_test.get_results()
if results.statistical_significance > 0.95:
    winner = results.winning_model
    print(f"Winner: {winner} with {results.improvement:.2%} improvement")
```

## 🔍 MLFlow UI Integration

### Custom Dashboard Views
```python
# Create custom MLFlow dashboard
from opto.mlflow import Dashboard

dashboard = Dashboard("Agent Optimization Dashboard")

# Add custom charts
dashboard.add_chart(
    type="line_chart",
    title="Optimization Progress",
    x_axis="iteration",
    y_axis="score",
    experiments=["beam_search", "ucb_search"]
)

dashboard.add_chart(
    type="bar_chart", 
    title="Algorithm Comparison",
    x_axis="algorithm",
    y_axis="final_score"
)

# Launch dashboard
dashboard.serve(port=8080)
```

### Export and Reporting
```python
# Export experiment results
from opto.mlflow import ExperimentExporter

exporter = ExperimentExporter()

# Export to various formats
exporter.to_csv("experiment_results.csv")
exporter.to_json("experiment_results.json")
exporter.to_pdf("experiment_report.pdf")

# Generate automated reports
report = exporter.generate_report(
    template="optimization_summary",
    experiments=["beam_search", "ucb_search", "random_search"]
)
```

## 🎯 Best Practices

1. **Organize Experiments**: Use meaningful experiment names and tags
2. **Version Everything**: Track code, data, and model versions
3. **Document Thoroughly**: Add descriptions and metadata to runs
4. **Monitor Performance**: Set up alerts for production models
5. **Clean Up**: Regularly archive old experiments and models

## 📚 Integration Examples

### Research Workflow
```python
# Complete research workflow with MLFlow
import opto
import mlflow

def research_pipeline():
    mlflow.set_experiment("research-agent-optimization")
    
    # Data preparation
    with mlflow.start_run(run_name="data_prep"):
        train_data, val_data = prepare_data()
        mlflow.log_param("train_size", len(train_data))
        mlflow.log_param("val_size", len(val_data))
        
    # Model training and optimization
    with mlflow.start_run(run_name="optimization"):
        trainer = opto.Trainer(
            algorithm=BeamSearchAlgorithm(beam_size=10),
            evaluator=MultiMetricEvaluator()
        )
        
        result = trainer.fit(train_data, val_data)
        
        # Log results
        mlflow.log_metric("best_score", result.best_score)
        mlflow.log_metric("training_time", result.training_time)
        mlflow.log_artifact(result.optimization_trace, "trace.json")
        
        # Register best model
        mlflow.sklearn.log_model(
            result.best_model,
            "optimized_agent",
            registered_model_name="research-agent"
        )
```

## 📚 Learn More

- [MLFlow Integration API Reference](../api/features/mlflow/index.md)
- [MLFlow Autolog Documentation](../api/features/mlflow/autolog.md) 
- [Model Registry Guide](../tutorials/trainers.ipynb)
- [Production Deployment Tutorial](../tutorials/optimization_tutorial.ipynb)

Ready to track your optimization experiments? Check out the [trainer tutorial](../tutorials/trainers.ipynb) to see MLFlow integration in action!