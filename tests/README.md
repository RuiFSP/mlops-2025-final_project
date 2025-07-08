# Tests Directory

This directory contains all test files organized by testing level and purpose.

## 📁 Directory Structure

```
tests/
├── unit/                     # Unit tests for individual components
│   ├── simulation/           # Unit tests for simulation components
│   ├── test_data_loader.py   # Data loading tests
│   ├── test_monitoring.py    # Monitoring tests
│   └── test_trainer.py       # Model training tests
├── integration/              # Integration tests for component workflows
│   └── test_simulation_workflow.py  # Full simulation workflow tests
├── e2e/                      # End-to-end tests (placeholder)
└── fixtures/                 # Test fixtures and sample data
```

## 🧪 Test Types

### Unit Tests (`unit/`)
Test individual components in isolation.

| Test File | Purpose | Coverage |
|-----------|---------|----------|
| `test_data_loader.py` | Data loading functionality | Data preprocessing pipeline |
| `test_trainer.py` | Model training logic | Training algorithms and validation |
| `test_monitoring.py` | Monitoring components | Performance tracking, alerting |
| `simulation/` | Simulation engine components | Match scheduling, odds generation, etc. |

### Integration Tests (`integration/`)
Test component interactions and workflows.

| Test File | Purpose | Coverage |
|-----------|---------|----------|
| `test_simulation_workflow.py` | Full simulation pipeline | End-to-end simulation process |

### End-to-End Tests (`e2e/`)
Test complete system workflows (planned for future development).

### Fixtures (`fixtures/`)
Shared test data and mock objects for consistent testing.

## 🚀 Running Tests

### All Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run with verbose output
pytest -v
```

### Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Specific test file
pytest tests/unit/test_data_loader.py

# Specific test function
pytest tests/unit/test_trainer.py::test_model_training
```

### Test Selection
```bash
# Run tests matching a pattern
pytest -k "simulation"

# Run tests with specific marker
pytest -m "slow"

# Skip specific tests
pytest -k "not slow"
```

## 📊 Test Coverage

The project maintains high test coverage across all components:

- **Unit Tests**: Core component functionality
- **Integration Tests**: Component interaction workflows
- **Mock Data**: Realistic test fixtures for consistent testing
- **Performance Tests**: Monitoring and alerting validation

## 🔧 Test Configuration

Test configuration is managed through:

- `pytest.ini` or `pyproject.toml` - Test runner configuration
- `conftest.py` - Shared fixtures and test setup
- Test markers for categorizing test types

## 📝 Adding New Tests

When adding new tests, follow these guidelines:

1. **Choose the right directory** based on test scope
2. **Use descriptive test names** that explain what's being tested
3. **Include docstrings** for complex test logic
4. **Use fixtures** for shared test data
5. **Mock external dependencies** for unit tests
6. **Update this README** when adding new test categories

## 🔗 Related Documentation

- `scripts/README.md` - Scripts organization and usage
- Main `README.md` - Project overview and setup
- CI/CD workflows - Automated testing configuration
