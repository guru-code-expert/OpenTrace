#!/usr/bin/env python3
"""
Example demonstrating how to use trace.mlflow.autolog() functionality.

This example shows how users can enable MLflow autologging for Trace operations
by calling trace.mlflow.autolog() at the beginning of their scripts.

The MLflow feature is organized in the features/mlflow/ directory but can be
accessed through the trace.mlflow namespace for a clean API.
"""

import opto.trace as trace

# Enable MLflow autologging - this should be called at the beginning of your script
trace.mlflow.autolog()

# You can also enable with custom parameters
# trace.mlflow.autolog(
#     log_models=True,
#     log_datasets=True,
#     registered_model_name="my_trace_model",
#     extra_tags={"experiment": "trace_optimization", "version": "1.0"}
# )

print("MLflow autologging enabled!")
print(f"Autolog status: {trace.mlflow.is_autolog_enabled()}")
print(f"Autolog config: {trace.mlflow.get_autolog_config()}")

# Your Trace operations here will now be automatically logged to MLflow
# For example:
@trace.node
def example_function(x):
    return x * 2

# When you run Trace operations, they will be automatically logged to MLflow
# if MLflow is properly configured in your environment

# You can check the global setting anywhere in your code
if trace.settings.mlflow_autologging:
    print("MLflow autologging is active - operations will be logged!")

# To disable autologging later:
# trace.mlflow.disable_autolog()

