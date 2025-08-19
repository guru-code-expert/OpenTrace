#!/usr/bin/env python3
"""
MLflow Integration Tests for Trace Library

This test file provides comprehensive testing of MLflow integration with the Trace library.
It supports two modes:

1. Local Development Mode:
   - Automatically starts MLflow server on http://127.0.0.1:5000
   - Keeps server running for manual examination of results
   - Provides interactive menu to run different test scenarios
   - Use Ctrl+C to gracefully shutdown

2. CI/CD Mode:
   - Detects CI environment (GitHub Actions, etc.)
   - Runs tests without starting server
   - Uses file-based tracking for validation
   - Exits automatically after tests complete

Usage:
    Local: python tests/mlflow_test.py
    CI:    pytest tests/mlflow_test.py
"""

import os
import sys
import time
import signal
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional
import atexit

import mlflow
from mlflow.entities import SpanType
import opto.trace as trace


class MLflowTestEnvironment:
    """Manages MLflow test environment setup and teardown."""
    
    def __init__(self):
        self.server_process: Optional[subprocess.Popen] = None
        self.temp_dir: Optional[str] = None
        self.is_ci = self._detect_ci_environment()
        self.original_tracking_uri = None
        
    def _detect_ci_environment(self) -> bool:
        """Detect if running in CI environment."""
        ci_indicators = [
            'CI', 'CONTINUOUS_INTEGRATION', 'GITHUB_ACTIONS', 
            'TRAVIS', 'CIRCLECI', 'JENKINS_URL', 'GITLAB_CI'
        ]
        return any(os.getenv(indicator) for indicator in ci_indicators)
    
    def setup(self):
        """Set up the appropriate MLflow environment."""
        if self.is_ci:
            self._setup_ci_environment()
        else:
            self._setup_local_environment()
            
    def _setup_ci_environment(self):
        """Set up file-based tracking for CI."""
        print("🔧 Setting up CI environment with file-based tracking...")
        self.temp_dir = tempfile.mkdtemp(prefix="mlflow_test_")
        tracking_uri = f"file://{self.temp_dir}/mlruns"
        
        self.original_tracking_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("trace-ci-test")
        
        print(f"📁 Using temporary MLflow tracking: {tracking_uri}")
        
    def _setup_local_environment(self):
        """Set up MLflow server for local development."""
        print("🚀 Setting up local development environment...")
        
        # Create local mlruns directory
        os.makedirs("mlruns", exist_ok=True)
        
        # Start MLflow server
        print("🔄 Starting MLflow server on http://127.0.0.1:5000...")
        try:
            self.server_process = subprocess.Popen([
                "mlflow", "server",
                "--host", "127.0.0.1",
                "--port", "5000",
                "--backend-store-uri", "sqlite:///mlruns.db"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for server to start
            print("⏳ Waiting for MLflow server to start...")
            time.sleep(3)
            
            # Test connection
            if self._test_server_connection():
                print("✅ MLflow server started successfully!")
                mlflow.set_tracking_uri("http://127.0.0.1:5000")
                mlflow.set_experiment("trace-local-test")
            else:
                raise Exception("Failed to connect to MLflow server")
                
        except Exception as e:
            print(f"❌ Failed to start MLflow server: {e}")
            print("💡 Make sure MLflow is installed: pip install mlflow")
            sys.exit(1)
            
    def _test_server_connection(self, max_retries=10) -> bool:
        """Test if MLflow server is responding."""
        try:
            import requests
        except ImportError:
            print("⚠️  Warning: requests library not available for server health check")
            return True  # Assume server is running if we can't check
        
        for i in range(max_retries):
            try:
                response = requests.get("http://127.0.0.1:5000/health", timeout=2)
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(1)
        return False
    
    def cleanup(self):
        """Clean up resources."""
        if self.server_process:
            print("\n🔄 Shutting down MLflow server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            print("✅ MLflow server stopped")
            
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"🗑️  Cleaned up temporary directory: {self.temp_dir}")
            
        if self.original_tracking_uri:
            mlflow.set_tracking_uri(self.original_tracking_uri)


# Global test environment instance
test_env = MLflowTestEnvironment()

def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        test_env.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# Register cleanup function
atexit.register(test_env.cleanup)


# Test Functions
@trace.bundle(mlflow_kwargs={'span_type': SpanType.LLM, 'silent': True})
def nested_silent_func(x):
    """A nested function with silent MLflow logging."""
    return x + 1

@trace.bundle(mlflow_kwargs={'span_type': SpanType.LLM})
def nested_func_2(x):
    """A nested function with MLflow logging."""
    return x * 2

@trace.bundle(mlflow_kwargs={'span_type': SpanType.LLM})
def my_complex_function(x, y):
    """A complex function that calls other traced functions."""
    return nested_silent_func(nested_func_2(x + y))

@trace.bundle(mlflow_kwargs={'span_type': SpanType.CHAIN})
def chain_function(items):
    """A function that processes a chain of items."""
    result = []
    for item in items:
        processed = nested_func_2(item)
        result.append(processed)
    return result


def test_autolog_api():
    """Test the MLflow autolog API functionality."""
    print("🧪 Testing MLflow autolog API...")
    
    # Test initial state
    initial_state = trace.mlflow.is_autolog_enabled()
    print(f"   Initial autolog state: {initial_state}")
    
    # Enable autolog
    trace.mlflow.autolog()
    enabled_state = trace.mlflow.is_autolog_enabled()
    print(f"   After enabling autolog: {enabled_state}")
    assert enabled_state is True, "Autolog should be enabled"
    
    # Test configuration retrieval
    config = trace.mlflow.get_autolog_config()
    print(f"   Autolog config: {config}")
    assert isinstance(config, dict), "Config should be a dictionary"
    
    # Test disable
    trace.mlflow.disable_autolog()
    disabled_state = trace.mlflow.is_autolog_enabled()
    print(f"   After disabling autolog: {disabled_state}")
    assert disabled_state is False, "Autolog should be disabled"
    
    # Re-enable for other tests
    trace.mlflow.autolog()
    print("✅ MLflow autolog API tests passed!")


def test_nested_functions():
    """Test nested function tracing."""
    print("🧪 Testing nested function tracing...")
    
    with mlflow.start_run(run_name="nested_functions_test"):
        result = my_complex_function(5, 3)
        expected = nested_silent_func(nested_func_2(5 + 3))  # (5+3)*2 + 1 = 17
        
        print(f"   Input: x=5, y=3")
        print(f"   Result: {result}")
        print(f"   Expected: {expected}")
        
        assert result == expected, f"Expected {expected}, got {result}"
    
    print("✅ Nested function tracing tests passed!")


def test_chain_processing():
    """Test chain processing with MLflow tracing."""
    print("🧪 Testing chain processing...")
    
    with mlflow.start_run(run_name="chain_processing_test"):
        items = [1, 2, 3, 4, 5]
        result = chain_function(items)
        expected = [item * 2 for item in items]
        
        print(f"   Input items: {items}")
        print(f"   Result: {result}")
        print(f"   Expected: {expected}")
        
        assert result == expected, f"Expected {expected}, got {result}"
    
    print("✅ Chain processing tests passed!")


def test_error_handling():
    """Test error handling in traced functions."""
    print("🧪 Testing error handling...")
    
    @trace.bundle(mlflow_kwargs={'span_type': SpanType.LLM})
    def error_function():
        raise ValueError("Test error for MLflow tracing")
    
    with mlflow.start_run(run_name="error_handling_test"):
        try:
            error_function()
            assert False, "Should have raised an error"
        except ValueError as e:
            print(f"   Caught expected error: {e}")
            assert str(e) == "Test error for MLflow tracing"
    
    print("✅ Error handling tests passed!")


def run_all_tests():
    """Run all test functions."""
    print("🔬 Running all MLflow integration tests...\n")
    
    try:
        test_autolog_api()
        print()
        
        test_nested_functions()
        print()
        
        test_chain_processing()
        print()
        
        test_error_handling()
        print()
        
        print("🎉 All tests passed successfully!")
        
        if not test_env.is_ci:
            print(f"\n🌐 MLflow UI available at: http://127.0.0.1:5000")
            print("📊 Check the UI to see logged traces and experiments")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


def interactive_menu():
    """Display interactive menu for local testing."""
    while True:
        print("\n" + "="*50)
        print("🧪 MLflow Integration Test Menu")
        print("="*50)
        print("1. Run all tests")
        print("2. Test autolog API")
        print("3. Test nested functions")
        print("4. Test chain processing")
        print("5. Test error handling")
        print("6. Open MLflow UI (in browser)")
        print("0. Exit")
        print("-" * 50)
        
        try:
            choice = input("Select an option (0-6): ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                run_all_tests()
            elif choice == "2":
                test_autolog_api()
            elif choice == "3":
                test_nested_functions()
            elif choice == "4":
                test_chain_processing()
            elif choice == "5":
                test_error_handling()
            elif choice == "6":
                import webbrowser
                webbrowser.open("http://127.0.0.1:5000")
                print("🌐 Opening MLflow UI in browser...")
            else:
                print("❌ Invalid option. Please try again.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")


# Pytest-compatible test functions (for CI)
def test_mlflow_autolog_api():
    """Pytest-compatible autolog API test."""
    test_autolog_api()

def test_mlflow_nested_functions():
    """Pytest-compatible nested functions test."""
    test_nested_functions()

def test_mlflow_chain_processing():
    """Pytest-compatible chain processing test."""
    test_chain_processing()

def test_mlflow_error_handling():
    """Pytest-compatible error handling test."""
    test_error_handling()


def main():
    """Main entry point."""
    print("🔍 MLflow Integration Tests for Trace Library")
    print(f"🏃 Running in {'CI' if test_env.is_ci else 'LOCAL'} mode")
    
    # Set up environment
    setup_signal_handlers()
    test_env.setup()
    
    # Enable autolog
    trace.mlflow.autolog()
    
    if test_env.is_ci:
        # Run all tests in CI mode
        print("🤖 Running automated tests...")
        run_all_tests()
        print("✅ CI tests completed successfully!")
    else:
        # Interactive mode for local development
        print("👨‍💻 Starting interactive test menu...")
        print("💡 Use Ctrl+C to exit gracefully")
        
        try:
            interactive_menu()
        except KeyboardInterrupt:
            pass
        
        print("\n👋 Thanks for testing! MLflow server will remain running.")
        print("🌐 MLflow UI: http://127.0.0.1:5000")
        print("🛑 Use Ctrl+C again to stop the server")
        
        # Keep server running until user decides to stop
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping MLflow server...")


if __name__ == "__main__":
    main()