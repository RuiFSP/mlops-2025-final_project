# Scripts Directory

This directory contains executable scripts organized by domain/functionality.

## 📁 Directory Structure

```
scripts/
├── data/                     # Data collection and preparation
├── simulation/               # Season simulation scripts
├── monitoring/               # Monitoring and observability
├── testing/                  # Manual testing scripts
├── utils/                    # Utility and maintenance scripts
└── README.md                # This file
```

## 📊 Data Scripts (`data/`)
Scripts for data collection, preparation, and processing.

| Script | Purpose | Usage |
|--------|---------|-------|
| `collect_real_data.py` | Collect real Premier League data | `python scripts/data/collect_real_data.py` |
| `prepare_simulation_data.py` | Prepare data for season simulation | `python scripts/data/prepare_simulation_data.py` |

## 🏟️ Simulation Scripts (`simulation/`)
Scripts for running the Premier League season simulation engine.

| Script | Purpose | Usage |
|--------|---------|-------|
| `demo_simulation.py` | Quick 3-week simulation demo | `python scripts/simulation/demo_simulation.py` |
| `run_simulation.py` | Full season simulation runner | `python scripts/simulation/run_simulation.py --mode batch` |

### Simulation Examples
```bash
# Quick demo
python scripts/simulation/demo_simulation.py

# Interactive simulation
python scripts/simulation/run_simulation.py --mode interactive --weeks 10

# Full season
python scripts/simulation/run_simulation.py --mode batch

# Custom range
python scripts/simulation/run_simulation.py --start-week 5 --weeks 15
```

## 📈 Monitoring Scripts (`monitoring/`)
Scripts for monitoring model performance and system health.

| Script | Purpose | Usage |
|--------|---------|-------|
| `demo_monitoring.py` | Monitoring system demonstration | `python scripts/monitoring/demo_monitoring.py` |

## 🧪 Testing Scripts (`testing/`)
Manual testing scripts for API and model validation (not unit tests).

| Script | Purpose | Usage |
|--------|---------|-------|
| `test_enhanced_api.py` | Manual API testing and validation | `python tests/e2e/test_enhanced_api.py` |
| `test_enhanced_model.py` | Manual model testing and evaluation | `python tests/e2e/test_enhanced_model.py` |

**Note:** These are integration/manual testing scripts. Unit tests are in `tests/unit/`.

## 🔧 Utility Scripts (`utils/`)
General utility and maintenance scripts.

| Script | Purpose | Usage |
|--------|---------|-------|
| `project_status.py` | Project status and health check | `python scripts/utils/project_status.py` |
| `run_checks.py` | Quality and dependency checks | `python scripts/utils/run_checks.py` |

## 🚀 Common Usage Patterns

### Development Workflow
```bash
# 1. Check project status
python scripts/utils/project_status.py

# 2. Prepare data (if needed)
python scripts/data/prepare_simulation_data.py

# 3. Run simulation demo
python scripts/simulation/demo_simulation.py

# 4. Run quality checks
python scripts/utils/run_checks.py
```

### Production Workflow
```bash
# 1. Collect fresh data
python scripts/data/collect_real_data.py

# 2. Run full season simulation
python scripts/simulation/run_simulation.py --mode batch

# 3. Monitor performance
python scripts/monitoring/demo_monitoring.py
```

## 📝 Adding New Scripts

When adding new scripts, follow these guidelines:

1. **Choose the right directory** based on script purpose
2. **Add executable permissions**: `chmod +x script_name.py`
3. **Include proper shebang**: `#!/usr/bin/env python3`
4. **Add docstring** with description and usage
5. **Update this README** with the new script information

## 🔗 Related Documentation

- `tests/README.md` - Testing strategy and structure
- Main `README.md` - Project overview and setup
- `src/` documentation - Core library components
