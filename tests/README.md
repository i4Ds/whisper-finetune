# Tests for Whisper Fine-tuning

This directory contains unit tests for the whisper-finetune package.

## Running Tests

### Prerequisites

First, make sure you have the development dependencies installed:

```bash
# Activate your conda environment
conda activate whisper-finetune

# Install the package with dev dependencies
pip install -e ".[dev]"
```

### Run All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=whisper_finetune --cov-report=html

# Run specific test file
pytest tests/test_metrics.py

# Run specific test class
pytest tests/test_metrics.py::TestWERComputation

# Run specific test
pytest tests/test_metrics.py::TestWERComputation::test_perfect_match
```

### Test Organization

- `test_metrics.py` - Tests for evaluation metrics (WER, CER, NLL, entropy, ECE, etc.)
- `test_utils.py` - Tests for text normalization and vocabulary utilities
- `test_training_utils.py` - Tests for training utilities (seed setting, step calculation, etc.)

### Continuous Integration

These tests are designed to be run in CI/CD pipelines. They are:
- **Fast**: Run in seconds, no heavy model loading
- **Isolated**: No external dependencies or network calls
- **Deterministic**: Use fixed seeds for reproducibility

### Writing New Tests

When adding new functionality:

1. Create tests in the appropriate test file (or create a new one)
2. Follow the existing naming convention: `test_<functionality>`
3. Use descriptive test names that explain what is being tested
4. Include docstrings explaining the test purpose
5. Use pytest fixtures for common setup/teardown

Example:
```python
def test_my_new_feature():
    """Test that my new feature works correctly."""
    result = my_function(input_data)
    assert result == expected_output
```

### Test Markers

Some tests can be marked for special handling:

```python
@pytest.mark.slow
def test_expensive_operation():
    """Test that takes a long time."""
    pass

# Skip slow tests with:
pytest -m "not slow"
```

### Debugging Failed Tests

```bash
# Show full output for failed tests
pytest --tb=long

# Stop at first failure
pytest -x

# Enter debugger on failure
pytest --pdb
```
