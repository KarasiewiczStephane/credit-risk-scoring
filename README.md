# Credit Risk Scoring Model

> Interpretable ML model for loan approval decisions with fairness testing and regulatory compliance.

[![CI](https://github.com/KarasiewiczStephane/credit-risk-scoring/actions/workflows/ci.yml/badge.svg)](https://github.com/KarasiewiczStephane/credit-risk-scoring/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project implements an interpretable credit risk scoring system using LightGBM with:

- **Traditional Scorecard**: WoE/IV analysis with point allocation (PDO=20, base score 600)
- **Explainability**: SHAP TreeExplainer, LIME, and Partial Dependence Plots
- **Fairness Testing**: Demographic parity, equalized odds analysis with bias mitigation
- **Regulatory Compliance**: Model card following Google's format, ECOA/FCRA checklist
- **Production API**: FastAPI with health checks, Docker containerization, CI pipeline

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                       │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────┤
│  /score     │  /explain   │ /fairness   │ /model-card │ /health │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┴─────────┘
       │             │             │             │
       ▼             ▼             ▼             ▼
┌──────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
│  Scorecard   │ │   SHAP   │ │ Fairlearn│ │  Model Card  │
│  LightGBM    │ │   LIME   │ │ Analyzer │ │  Generator   │
└──────────────┘ └──────────┘ └──────────┘ └──────────────┘
       │             │             │             │
       └─────────────┴──────┬──────┴─────────────┘
                            ▼
                    ┌──────────────┐
                    │  Data Layer  │
                    │  WoE/IV      │
                    │  Preprocessor│
                    └──────────────┘
```

## Quick Start

```bash
# Clone repository
git clone https://github.com/KarasiewiczStephane/credit-risk-scoring.git
cd credit-risk-scoring

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
make test

# Start API server
make run
```

## Scorecard Example

### Traditional Credit Scorecard

The scorecard converts LightGBM predictions into interpretable credit scores using Weight of Evidence (WoE) binning and logistic regression point allocation.

| Feature | Bin | WoE | Points |
|---------|-----|-----|--------|
| checking_status | No account (A14) | 0.52 | +38 |
| checking_status | < 0 DM (A11) | -0.84 | -61 |
| credit_history | Critical (A30) | -0.89 | -65 |
| credit_history | Existing paid (A34) | 0.43 | +31 |
| duration | 0-12 months | 0.28 | +20 |
| duration | 36+ months | -0.45 | -33 |

**Base Score**: 600 | **PDO**: 20 | **Score Range**: 350-850

### Score Interpretation

| Score Range | Risk Band | PD Range | Recommendation |
|-------------|-----------|----------|----------------|
| 750+ | Excellent | < 2% | Auto-approve |
| 700-749 | Good | 2-5% | Approve |
| 650-699 | Fair | 5-10% | Manual review |
| 600-649 | Poor | 10-20% | Conditional |
| < 600 | Very Poor | > 20% | Decline |

## Model Performance

| Metric | LightGBM | Logistic Regression | Random Forest |
|--------|----------|---------------------|---------------|
| AUC-ROC | **0.78** | 0.75 | 0.76 |
| Gini | 0.56 | 0.50 | 0.52 |
| KS Statistic | 0.42 | 0.38 | 0.40 |
| Accuracy | 0.77 | 0.74 | 0.75 |

## Fairness Analysis

### Before Mitigation

| Protected Attribute | Demographic Parity Diff | Equalized Odds Diff |
|---------------------|------------------------|---------------------|
| Age Group | 0.15 | 0.12 |
| Gender (from status) | 0.08 | 0.10 |

### After Mitigation (Threshold Optimization)

| Protected Attribute | Demographic Parity Diff | Equalized Odds Diff |
|---------------------|------------------------|---------------------|
| Age Group | **0.04** | **0.05** |
| Gender (from status) | **0.03** | **0.04** |

Bias mitigation strategies implemented:
- **Reweighting** (pre-processing): Adjusts sample weights to balance group positive rates
- **Threshold Optimization** (post-processing): Per-group decision thresholds via Fairlearn
- **Exponentiated Gradient** (in-processing): Constrained optimization with demographic parity or equalized odds

## API Usage

### Score a Credit Application

```bash
curl -X POST "http://localhost:8000/score" \
  -H "Content-Type: application/json" \
  -d '{
    "application_id": "APP-001",
    "checking_status": "A11",
    "duration": 24,
    "credit_history": "A34",
    "purpose": "A43",
    "credit_amount": 5000,
    "savings_status": "A61",
    "employment": "A73",
    "installment_rate": 2,
    "personal_status": "A93",
    "other_parties": "A101",
    "residence_since": 2,
    "property_magnitude": "A121",
    "age": 35,
    "other_payment_plans": "A143",
    "housing": "A152",
    "existing_credits": 1,
    "job": "A173",
    "num_dependents": 1,
    "own_telephone": "A192",
    "foreign_worker": "A201"
  }'
```

Response:

```json
{
  "application_id": "APP-001",
  "credit_score": 685,
  "probability_of_default": 0.0821,
  "decision": "approved",
  "score_band": "Fair",
  "explanation_summary": {
    "top_factors": [
      ["checking_status", -0.15],
      ["duration", 0.08],
      ["credit_amount", -0.05]
    ]
  },
  "processing_time_ms": 45.2
}
```

### Get Detailed Explanation

```bash
curl "http://localhost:8000/explain/APP-001"
```

### Get Fairness Report

```bash
curl "http://localhost:8000/fairness-report"
```

### Get Model Card

```bash
curl "http://localhost:8000/model-card"
```

## Docker

```bash
# Build image
make docker-build

# Run container
make docker-run

# Or use docker compose
make docker-compose-up
```

## Project Structure

```
credit-risk-scoring/
├── src/
│   ├── api/               # FastAPI endpoints and Pydantic schemas
│   ├── data/              # Data loading, preprocessing, WoE/IV
│   ├── models/            # Training, scorecard, evaluation, model card
│   ├── explainability/    # SHAP, LIME, PDP generators
│   ├── fairness/          # Fairlearn analysis, bias mitigation
│   └── utils/             # Configuration, structured logging
├── tests/                 # Unit and integration tests (220+)
├── configs/               # YAML configuration
├── data/sample/           # Sample data for CI testing
├── .github/workflows/     # CI/CD pipeline
├── Dockerfile             # Multi-stage production build
├── docker-compose.yml     # Local development
├── Makefile               # Build automation
└── requirements.txt       # Python dependencies
```

## Regulatory Compliance

The model card includes ECOA and FCRA compliance checklists:

**ECOA (Equal Credit Opportunity Act)**:
- No prohibited factors (race, color, religion, national origin, sex, marital status)
- Adverse action notice with specific reasons
- Disparate impact analysis on protected groups

**FCRA (Fair Credit Reporting Act)**:
- Permissible purpose verification
- Consumer disclosure capability
- Dispute resolution process
- Accuracy assurance procedures

## Development

```bash
# Install dev dependencies
pip install -r requirements.txt
pip install -e .

# Run linter
make lint

# Run tests with coverage
make test

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

## License

MIT
