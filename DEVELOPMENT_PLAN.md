# Premier League Match Predictor - Development Plan

## 🎯 **DEVELOPMENT PHILOSOPHY**
- **Local First**: All development happens locally
- **Full Stack**: Keep all MLOps components (PostgreSQL, MLflow, Prefect, EvidentlyAI, Grafana)
- **uv Management**: Modern Python package management
- **Future Ready**: Easy migration to Docker/cloud later

---

## 📁 **PROJECT STRUCTURE**
```
mlops-2025-final_project/
├── src/                          # Core source code
│   ├── pipelines/               # Training & prediction pipelines
│   ├── betting_simulator/       # Betting simulation logic
│   ├── monitoring/              # EvidentlyAI monitoring
│   └── main.py                  # Main entry point
├── scripts/                     # Development & testing scripts
├── data/                        # Training data (3,040 matches)
├── models/                      # Local model storage
├── .env                         # Local environment variables
├── pyproject.toml              # Full MLOps stack dependencies
└── README.md                   # Local development guide
```

---

## 🏗️ **FULL MLOPS STACK (Local Development)**
- ✅ **PostgreSQL** - Data storage & persistence
- ✅ **MLflow** - Model tracking & registry
- ✅ **Prefect** - Workflow orchestration
- ✅ **EvidentlyAI** - ML monitoring & drift detection
- ✅ **Grafana** - Dashboards & visualization
- ✅ **Scikit-learn** - ML model (Random Forest)
- ✅ **Pandas** - Data processing
- ✅ **FastAPI** - Future REST API

---

## 📋 **DEVELOPMENT PHASES**

### **Phase 1: Local Development Setup** 🔴 **HIGH PRIORITY**

#### **1.1 Environment Setup**
- [ ] **Install Dependencies**
  ```bash
  uv sync
  ```
- [ ] **Configure Environment**
  ```bash
  cp config.env.example .env
  # Edit .env with local settings
  ```
- [ ] **Verify .env Configuration**
  - PostgreSQL: localhost:5432
  - MLflow: http://127.0.0.1:5000
  - All other services configured

#### **1.2 Start Local Services**
- [ ] **PostgreSQL Server**
  - Status: Running on localhost:5432
  - Database: mlops_db
  - User: mlops_user
- [ ] **MLflow Server**
  ```bash
  mlflow server --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
  ```
- [ ] **Prefect Server**
  ```bash
  prefect server start
  ```
- [ ] **Grafana** (Optional for Phase 1)
  - Status: Running on localhost:3000
  - Credentials: admin/admin

#### **1.3 Database Setup**
- [ ] **Initialize Database Schema**
  ```bash
  uv run python scripts/setup_database.py
  ```
- [ ] **Verify Tables Created**
  ```bash
  uv run python scripts/check_db_tables.py
  ```
- [ ] **Clean Database (if needed)**
  ```bash
  uv run python scripts/clean_postgres.py
  ```

#### **1.4 Core Pipeline Testing**
- [ ] **Training Pipeline**
  ```bash
  uv run python -m src.pipelines.training_pipeline
  ```
  - Expected: Model trained and registered in MLflow
  - Expected: Accuracy ~61.84%
- [ ] **Prediction Pipeline**
  ```bash
  uv run python -m src.pipelines.prediction_pipeline
  ```
  - Expected: 5 predictions generated
- [ ] **Betting Simulation**
  ```bash
  uv run python scripts/test_betting_simulation.py
  ```
  - Expected: Bets placed and saved to database
- [ ] **Verify Database State**
  ```bash
  uv run python scripts/debug_bets_table.py
  ```
  - Expected: Bets visible in database

#### **1.5 Service Verification**
- [ ] **MLflow UI**: http://127.0.0.1:5000
  - Check experiments and registered models
- [ ] **Prefect UI**: http://127.0.0.1:4200
  - Verify workflow orchestration
- [ ] **PostgreSQL**: Verify data persistence
- [ ] **Grafana**: http://localhost:3000 (if running)

---

### **Phase 2: Core Functionality** 🟡 **MEDIUM PRIORITY**

#### **2.1 Production Betting Thresholds**
- [ ] **Update Betting Configuration**
  - Current: confidence: 0.35, margin: 0.05 (testing)
  - Target: confidence: 0.6, margin: 0.1 (production)
- [ ] **Test with Production Thresholds**
- [ ] **Verify Betting Performance**

#### **2.2 Real Data Integration**
- [ ] **Identify Data Source**
  - Current: Simulated matches
  - Target: Real Premier League data
- [ ] **Implement Data Pipeline**
- [ ] **Update Training Pipeline**
- [ ] **Test with Real Data**

#### **2.3 REST API Development**
- [ ] **FastAPI Setup**
- [ ] **Prediction Endpoints**
- [ ] **Betting Simulation Endpoints**
- [ ] **Authentication & Rate Limiting**
- [ ] **API Documentation**

#### **2.4 Automated Retraining**
- [ ] **Prefect Workflow Setup**
- [ ] **Performance Monitoring**
- [ ] **Retraining Triggers**
- [ ] **Model Versioning**

---

### **Phase 3: Advanced Features** 🟢 **LOW PRIORITY**

#### **3.1 Grafana Dashboards**
- [ ] **Model Performance Dashboard**
- [ ] **Betting Statistics Dashboard**
- [ ] **System Health Dashboard**
- [ ] **Real-time Monitoring**

#### **3.2 EvidentlyAI Monitoring**
- [ ] **Data Drift Detection**
- [ ] **Model Performance Monitoring**
- [ ] **Alert System**
- [ ] **Monitoring Dashboards**

#### **3.3 Testing Suite**
- [ ] **Unit Tests**
- [ ] **Integration Tests**
- [ ] **End-to-End Tests**
- [ ] **Test Coverage**

#### **3.4 Docker Migration**
- [ ] **Dockerfile Creation**
- [ ] **Docker Compose Setup**
- [ ] **Service Orchestration**
- [ ] **Production Deployment**

---

## 🚀 **EXECUTION COMMANDS**

### **Setup Commands**
```bash
# Install dependencies
uv sync

# Configure environment
cp config.env.example .env

# Start services
mlflow server --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
prefect server start
```

### **Core Pipeline Commands**
```bash
# Database setup
uv run python scripts/setup_database.py

# Training
uv run python -m src.pipelines.training_pipeline

# Prediction
uv run python -m src.pipelines.prediction_pipeline

# Betting simulation
uv run python scripts/test_betting_simulation.py

# Debug database
uv run python scripts/debug_bets_table.py
```

### **Testing Commands**
```bash
# Check database tables
uv run python scripts/check_db_tables.py

# Clean database
uv run python scripts/clean_postgres.py

# Test services
uv run python scripts/test_services.py
```

---

## 📊 **SUCCESS METRICS**

### **Phase 1 Success Criteria**
- [ ] All local services running
- [ ] Training pipeline completes successfully
- [ ] Model registered in MLflow with >60% accuracy
- [ ] Predictions generated successfully
- [ ] Bets placed and saved to database
- [ ] All components communicate properly

### **Phase 2 Success Criteria**
- [ ] Production betting thresholds implemented
- [ ] Real data integration working
- [ ] REST API functional
- [ ] Automated retraining pipeline operational

### **Phase 3 Success Criteria**
- [ ] Grafana dashboards operational
- [ ] EvidentlyAI monitoring active
- [ ] Comprehensive test coverage
- [ ] Docker deployment ready

---

## 🎯 **CURRENT STATUS**

**Phase**: 1 - Local Development Setup
**Status**: Ready to begin execution
**Next Action**: Start local services and test core pipeline

---

**Last Updated**: 2025-07-12
**Next Review**: After Phase 1 completion
