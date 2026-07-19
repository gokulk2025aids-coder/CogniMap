# CogniMap — Explainable Early-Warning System for Academic Risk & Performance Prediction

CogniMap is an end-to-end, production-structured machine learning platform that predicts student academic risk, final grades, and performance segments — and explains every prediction in plain English, with personalized intervention recommendations attached.

Built as a portfolio-grade demonstration of the **full ML engineering lifecycle**: data ingestion → cleaning → feature engineering → modeling → explainability → recommendation → API → dashboard, using two real, independent educational datasets.

---

## Table of Contents

- [Why This Project Exists](#why-this-project-exists)
- [Datasets](#datasets)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Feature Catalogue](#feature-catalogue-full-lifecycle)
- [Engineered Features](#engineered-features)
- [Machine Learning Modules](#machine-learning-modules)
- [Explainable AI](#explainable-ai)
- [Recommendation Engine](#recommendation-engine)
- [Backend API](#backend-api)
- [Database](#database)
- [Dashboard](#dashboard)
- [Installation](#installation)
- [Testing](#testing)
- [Project Structure](#project-structure)

---

## Why This Project Exists

Most student-performance-prediction projects on GitHub use a single small dataset (usually just the UCI Student Performance dataset) and stop at a single classifier with an accuracy score. EduRadar is deliberately broader:

- **Two real, independent datasets** (OULAD + UCI), used for the supervised tasks each one actually supports, rather than forced together.
- **A composite, validated "Academic Health Score"** that correlates monotonically with real outcomes — without ever seeing the label during construction.
- **Explicit leakage prevention**, including a caught case of engineered features that indirectly encoded their own target.
- **Explainability as a first-class feature**, not an afterthought — every prediction should be answerable with "why."
- **A recommendation layer** that turns predictions into concrete next actions for an advisor, not just a risk score.

---

## Datasets

### 1. Open University Learning Analytics Dataset (OULAD)
- 32,593 students across 7 relational tables, including a real VLE (Virtual Learning Environment) clickstream log (~10.6M interaction events).
- Provides genuine **behavioral** data: demographics, assessment scores, day-by-day engagement, and final outcome (`Withdrawn` / `Fail` / `Pass` / `Distinction`).
- Used for: **Risk Classification (Module 1)** and **Segmentation (Module 3)**, since it has real dropout/outcome labels and rich behavioral signal.

### 2. UCI Student Performance Dataset
- 1,044 records (395 Math + 649 Portuguese), with rich demographic/socio-economic survey data and three real grade checkpoints (G1, G2, G3, each 0-20).
- Used for: **Grade Regression (Module 2)**, since it has a genuine continuous target that OULAD lacks.

**Design principle:** these are treated as two separate pipelines feeding a common feature schema, not naively merged — different student populations, granularities, and targets mean row-level concatenation would create fake associations. Where a feature can't be measured the same way in both datasets (e.g. UCI has no clickstream), this is documented as a **proxy**, not silently treated as equivalent.

---

## Architecture

```
Raw Data (OULAD + UCI)
        │
        ▼
┌───────────────────┐
│  Preprocessing     │  cleaning, dedup, encoding (src/preprocessing/)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Feature Engineering│  7 core features, per-dataset (src/feature_engineering/)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   Machine Learning  │  risk / grade / segmentation models (src/models/)
└───────────────────┘
        │
        ▼
┌───────────────────┐        ┌───────────────────┐
│  Explainability     │──────▶│ Recommendation Eng. │
│  (src/explainability)│      │ (src/recommendation)│
└───────────────────┘        └───────────────────┘
        │                              │
        └──────────────┬───────────────┘
                        ▼
              ┌───────────────────┐
              │   FastAPI Backend  │  (api/)
              └───────────────────┘
                        │
        ┌───────────────┴───────────────┐
        ▼                                ▼
┌───────────────┐              ┌───────────────────┐
│  Database       │              │ Streamlit Dashboard│
│ (SQLite/Postgres)│◀────────────│    (dashboard/)     │
└───────────────┘              └───────────────────┘
```

`api/` and `dashboard/` are thin presentation layers — they import logic from `src/`, never reimplement it, so predictions/explanations/recommendations are computed identically everywhere they're used.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Backend | FastAPI, Pydantic, Uvicorn |
| Database | SQLAlchemy (SQLite dev / PostgreSQL prod) |
| Dashboard | Streamlit, Plotly |
| Testing | pytest, pytest-cov, httpx |
| Utilities | loguru (logging), joblib (model persistence) |

---

## Feature Catalogue (Full Lifecycle)

### Data Processing 
- Missing value handling (documented per-column strategy — never blanket mean/mode imputation)
- Duplicate removal (caught 787,170 duplicate clickstream rows in OULAD)
- Categorical encoding (ordinal domain-knowledge maps + one-hot for nominal fields)
- Class imbalance handling via SMOTE (applied leakage-safely, inside a pipeline fit only on training folds)
- Reusable, testable preprocessing pipelines (`BaseCleaner` abstract class, Template Method pattern)

### Feature Engineering 
7 named features built per-dataset (see [Engineered Features](#engineered-features) below), each with an explicit "direct measurement" or "proxy" classification.

### Machine Learning 
Three supervised/unsupervised modules (see [Machine Learning Modules](#machine-learning-modules) below), each using the dataset whose real target fits the task.

### Explainable AI 
- Global feature importance (SHAP summary plot, per risk class)
- Local, per-student explanation (top contributing features, signed direction)
- Waterfall plot per student
- Force plot per student
- Plain-English translation of SHAP output for non-technical advisors

### Recommendation Engine 
Rule-based, dynamically generated interventions mapped from feature thresholds, e.g.:
- Low attendance consistency → recommend mentor meeting
- Low LMS/engagement activity → recommend a daily structured learning schedule
- Declining quiz performance trend → recommend targeted revision sessions
- Poor assignment timeliness → recommend a deadline-management check-in

Recommendations will be generated per-student from the same feature table used for prediction, and will cite which feature(s) triggered which recommendation (so it's explainable, not a black box).

### Backend API 
FastAPI endpoints (planned contracts, finalized during Phase 7):

| Method | Endpoint | Description |
|---|---|---|
| GET | `/student/{id}` | Full profile: features, current predictions |
| POST | `/predict/risk` | Risk classification (Low/Medium/High) |
| POST | `/predict/grade` | Final grade regression |
| POST | `/predict/cluster` | Segment assignment (At Risk/Average/High Performer) |
| GET | `/explain/{id}` | SHAP explanation, plain-English + structured |
| GET | `/recommendation/{id}` | Personalized intervention list |

### Database 
Planned schema (SQLite for local dev, PostgreSQL-ready via SQLAlchemy):
- `students` — demographic + engineered feature snapshot
- `predictions` — risk/grade/cluster predictions with timestamps and model version
- `shap_explanations` — stored per-student SHAP contributions
- `recommendations` — generated interventions, linked to the feature(s) that triggered them

### Dashboard 
Streamlit multi-page app:
- **Home** — project overview, key stats
- **Dataset Analytics** — distributions, correlations, class balance
- **Risk Prediction** — per-student risk lookup + batch view
- **Grade Prediction** — regression results, actual-vs-predicted
- **Student Clusters** — segment visualization (PCA projection), segment profiles
- **Explainable AI** — SHAP summary + per-student waterfall, plain-English explanation
- **Recommendations** — intervention list per student/segment
- **Admin Panel** — model metrics, retraining trigger, data refresh status

### Documentation 
- Installation guide, user manual, API documentation (this README + `docs/`)
- Architecture diagram, ER diagram, flowchart (`docs/diagrams/`)

### GitHub Packaging 
- `.gitignore` , `requirements.txt` 
- Clean, phase-based commit history
- This README as the top-level project entry point

---

## Engineered Features

| Feature | OULAD (behavioral) | UCI (survey-based) |
|---|---|---|
| **Engagement Score** | Direct — total VLE clicks + active days | Proxy — `studytime` + `activities` |
| **Attendance Consistency** | Proxy — inverse coefficient of variation of weekly clicks | Direct — inverse of `absences` |
| **Assignment Timeliness** | Direct — days early/late vs. due date | Proxy (weak) — inverse of `failures` |
| **Quiz Performance Trend** | Direct — slope of assessment scores over time | Direct — slope across G1→G2→G3 |
| **LMS Activity Score** | Direct — diversity of VLE activity types used | Proxy (weakest) — `internet` access + `studytime` |
| **Study Efficiency** | Direct — score per unit of clicking effort | Direct — G3 per unit of `studytime` |
| **Academic Health Score** | Weighted composite of all above | Weighted composite (proxies downweighted) |

**Validation:** OULAD's Academic Health Score is monotonic with real outcomes — mean score rises Withdrawn (26.9) → Fail (39.6) → Pass (52.9) → Distinction (57.9) — despite never seeing the label during construction.

---

## Machine Learning Modules

### Module 1 — Academic Risk Prediction (OULAD) 
3-class classification: Low / Medium / High risk, mapped from `final_result` (Withdrawn→High, Fail→Medium, Pass/Distinction→Low).

| Model | Accuracy | F1 (macro) | ROC-AUC (OvR) |
|---|---|---|---|
| Logistic Regression | 0.738 | 0.687 | 0.881 |
| Random Forest | 0.770 | 0.718 | 0.903 |
| **XGBoost (best)** | **0.780** | **0.719** | **0.904** |

SMOTE applied inside an `imblearn.Pipeline`, fit only on the training fold, to address class imbalance (47% Low / 31% High / 22% Medium) without leaking test-set information.

### Module 2 — Final Grade Prediction (UCI) 
Regression on G3 (final grade, 0-20), deliberately **excluding G1/G2** as predictors (and excluding any Phase-3 feature derived from G3) to keep this a genuine early-warning problem rather than a near-tautological one.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Linear Regression (best)** | **2.62** | **3.71** | **0.111** |
| Random Forest | 2.63 | 3.74 | 0.097 |
| XGBoost | 2.66 | 3.74 | 0.096 |

### Module 3 — Student Segmentation (OULAD) 
K-Means (k=3, chosen via elbow + silhouette analysis across k=2-8) into **At Risk / Average / High Performer**, with cluster identity assigned post-hoc by ranking centroids on mean Academic Health Score (never fit on the label).

- Segment sizes: High Performer 17,362 · Average 11,865 · At Risk 3,366
- Validation: mean `final_result` rises monotonically across segments (0.11 → 0.88 → 1.73), confirming the clusters found real structure unsupervised.

---

## Explainable AI

Uses SHAP's `TreeExplainer` against the best-performing tree model (XGBoost) for exact, fast SHAP value computation. Produces:
- **Global summary plot** — which features matter most, per risk class
- **Local waterfall plot** — exactly how one student's prediction was built up, feature by feature
- **Force plot** — compact visual alternative to the waterfall
- **Plain-English translation** — e.g. *"This student is predicted as High Risk. Factors increasing this risk: low engagement score (12.4), low attendance consistency (34.1). Factors reducing this risk: high assignment timeliness (88.2)."*

The underlying scaler + classifier are extracted from the trained SMOTE pipeline before SHAP computation, since SHAP needs to explain the classifier's behavior on the same (scaled) input space it was trained on — SMOTE itself has no effect at inference time and is not part of the explanation.

---

## Recommendation Engine

Rule-based, per-student, generated dynamically from the engineered feature table (not hardcoded per student). Each rule fires off a specific feature crossing a documented threshold, and cites that feature back to the advisor so the recommendation is itself explainable:

```
Attendance Consistency < 40  →  "Recommend a mentor check-in — engagement has been irregular."
LMS Activity Score < 30      →  "Recommend a structured daily learning schedule."
Quiz Performance Trend < 0   →  "Recommend targeted revision sessions — recent scores are trending down."
Assignment Timeliness < 50   →  "Recommend a deadline-management conversation."
```

Multiple rules can fire for the same student; recommendations are ranked by how far the triggering feature is from its threshold (larger gap → higher priority).

---



## Database

Planned schema (SQLAlchemy models + raw SQL DDL will both be provided in `database/`):

```sql
CREATE TABLE students (
    student_key   VARCHAR PRIMARY KEY,   -- code_module|code_presentation|id_student
    dataset       VARCHAR,               -- 'oulad' or 'uci'
    features_json TEXT,                  -- engineered feature snapshot
    created_at    TIMESTAMP
);

CREATE TABLE predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    student_key   VARCHAR REFERENCES students(student_key),
    prediction_type VARCHAR,             -- 'risk' | 'grade' | 'cluster'
    prediction_value VARCHAR,
    model_name    VARCHAR,
    model_version VARCHAR,
    predicted_at  TIMESTAMP
);

CREATE TABLE shap_explanations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    student_key   VARCHAR REFERENCES students(student_key),
    feature_name  VARCHAR,
    shap_value    FLOAT,
    raw_value     FLOAT
);

CREATE TABLE recommendations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    student_key   VARCHAR REFERENCES students(student_key),
    recommendation TEXT,
    triggering_feature VARCHAR,
    priority      INTEGER,
    generated_at  TIMESTAMP
);
```

---

## Dashboard

Streamlit multi-page app (Home, Dataset Analytics, Risk Prediction, Grade Prediction, Student Clusters, Explainable AI, Recommendations, Admin Panel) — full build documented.

---

## Installation

```bash
git clone <repo-url>
cd EduRadar
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place raw data files (not committed to git) here:
```
data/raw/oulad/  # studentInfo.csv, studentRegistration.csv, studentAssessment.csv,
                 # assessments.csv, vle.csv, studentVle.csv, courses.csv
data/raw/uci/    # student-mat.csv, student-por.csv
```

## Testing

```bash
python -m pytest -m "not slow" -v      # fast unit tests
python -m pytest -m slow -v            # full integration tests against real data
python -m pytest --cov=src             # coverage report
```

## Project Structure

```
# CogniMap — Explainable Early-Warning System for Academic Risk & Performance Prediction

CogniMap is an end-to-end, production-structured machine learning platform that predicts student academic risk, final grades, and performance segments — and explains every prediction in plain English, with personalized intervention recommendations attached.

Built as a portfolio-grade demonstration of the **full ML engineering lifecycle**: data ingestion → cleaning → feature engineering → modeling → explainability → recommendation → API → dashboard, using two real, independent educational datasets.

---

## Table of Contents

- [Why This Project Exists](#why-this-project-exists)
- [Datasets](#datasets)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Feature Catalogue](#feature-catalogue-full-lifecycle)
- [Engineered Features](#engineered-features)
- [Machine Learning Modules](#machine-learning-modules)
- [Explainable AI](#explainable-ai)
- [Recommendation Engine](#recommendation-engine)
- [Backend API](#backend-api)
- [Database](#database)
- [Dashboard](#dashboard)
- [Installation](#installation)
- [Testing](#testing)
- [Project Structure](#project-structure)

---

## Why This Project Exists

Most student-performance-prediction projects on GitHub use a single small dataset (usually just the UCI Student Performance dataset) and stop at a single classifier with an accuracy score. EduRadar is deliberately broader:

- **Two real, independent datasets** (OULAD + UCI), used for the supervised tasks each one actually supports, rather than forced together.
- **A composite, validated "Academic Health Score"** that correlates monotonically with real outcomes — without ever seeing the label during construction.
- **Explicit leakage prevention**, including a caught case of engineered features that indirectly encoded their own target.
- **Explainability as a first-class feature**, not an afterthought — every prediction should be answerable with "why."
- **A recommendation layer** that turns predictions into concrete next actions for an advisor, not just a risk score.

---

## Datasets

### 1. Open University Learning Analytics Dataset (OULAD)
- 32,593 students across 7 relational tables, including a real VLE (Virtual Learning Environment) clickstream log (~10.6M interaction events).
- Provides genuine **behavioral** data: demographics, assessment scores, day-by-day engagement, and final outcome (`Withdrawn` / `Fail` / `Pass` / `Distinction`).
- Used for: **Risk Classification (Module 1)** and **Segmentation (Module 3)**, since it has real dropout/outcome labels and rich behavioral signal.

### 2. UCI Student Performance Dataset
- 1,044 records (395 Math + 649 Portuguese), with rich demographic/socio-economic survey data and three real grade checkpoints (G1, G2, G3, each 0-20).
- Used for: **Grade Regression (Module 2)**, since it has a genuine continuous target that OULAD lacks.

**Design principle:** these are treated as two separate pipelines feeding a common feature schema, not naively merged — different student populations, granularities, and targets mean row-level concatenation would create fake associations. Where a feature can't be measured the same way in both datasets (e.g. UCI has no clickstream), this is documented as a **proxy**, not silently treated as equivalent.

---

## Architecture

```
Raw Data (OULAD + UCI)
        │
        ▼
┌───────────────────┐
│  Preprocessing     │  cleaning, dedup, encoding (src/preprocessing/)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Feature Engineering│  7 core features, per-dataset (src/feature_engineering/)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   Machine Learning  │  risk / grade / segmentation models (src/models/)
└───────────────────┘
        │
        ▼
┌───────────────────┐        ┌───────────────────┐
│  Explainability     │──────▶│ Recommendation Eng. │
│  (src/explainability)│      │ (src/recommendation)│
└───────────────────┘        └───────────────────┘
        │                              │
        └──────────────┬───────────────┘
                        ▼
              ┌───────────────────┐
              │   FastAPI Backend  │  (api/)
              └───────────────────┘
                        │
        ┌───────────────┴───────────────┐
        ▼                                ▼
┌───────────────┐              ┌───────────────────┐
│  Database       │              │ Streamlit Dashboard│
│ (SQLite/Postgres)│◀────────────│    (dashboard/)     │
└───────────────┘              └───────────────────┘
```

`api/` and `dashboard/` are thin presentation layers — they import logic from `src/`, never reimplement it, so predictions/explanations/recommendations are computed identically everywhere they're used.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Backend | FastAPI, Pydantic, Uvicorn |
| Database | SQLAlchemy (SQLite dev / PostgreSQL prod) |
| Dashboard | Streamlit, Plotly |
| Testing | pytest, pytest-cov, httpx |
| Utilities | loguru (logging), joblib (model persistence) |

---

## Feature Catalogue (Full Lifecycle)

### Data Processing 
- Missing value handling (documented per-column strategy — never blanket mean/mode imputation)
- Duplicate removal (caught 787,170 duplicate clickstream rows in OULAD)
- Categorical encoding (ordinal domain-knowledge maps + one-hot for nominal fields)
- Class imbalance handling via SMOTE (applied leakage-safely, inside a pipeline fit only on training folds)
- Reusable, testable preprocessing pipelines (`BaseCleaner` abstract class, Template Method pattern)

### Feature Engineering 
7 named features built per-dataset (see [Engineered Features](#engineered-features) below), each with an explicit "direct measurement" or "proxy" classification.

### Machine Learning 
Three supervised/unsupervised modules (see [Machine Learning Modules](#machine-learning-modules) below), each using the dataset whose real target fits the task.

### Explainable AI 
- Global feature importance (SHAP summary plot, per risk class)
- Local, per-student explanation (top contributing features, signed direction)
- Waterfall plot per student
- Force plot per student
- Plain-English translation of SHAP output for non-technical advisors

### Recommendation Engine 
Rule-based, dynamically generated interventions mapped from feature thresholds, e.g.:
- Low attendance consistency → recommend mentor meeting
- Low LMS/engagement activity → recommend a daily structured learning schedule
- Declining quiz performance trend → recommend targeted revision sessions
- Poor assignment timeliness → recommend a deadline-management check-in

Recommendations will be generated per-student from the same feature table used for prediction, and will cite which feature(s) triggered which recommendation (so it's explainable, not a black box).

### Backend API 
FastAPI endpoints (planned contracts, finalized during Phase 7):

| Method | Endpoint | Description |
|---|---|---|
| GET | `/student/{id}` | Full profile: features, current predictions |
| POST | `/predict/risk` | Risk classification (Low/Medium/High) |
| POST | `/predict/grade` | Final grade regression |
| POST | `/predict/cluster` | Segment assignment (At Risk/Average/High Performer) |
| GET | `/explain/{id}` | SHAP explanation, plain-English + structured |
| GET | `/recommendation/{id}` | Personalized intervention list |

### Database 
Planned schema (SQLite for local dev, PostgreSQL-ready via SQLAlchemy):
- `students` — demographic + engineered feature snapshot
- `predictions` — risk/grade/cluster predictions with timestamps and model version
- `shap_explanations` — stored per-student SHAP contributions
- `recommendations` — generated interventions, linked to the feature(s) that triggered them

### Dashboard 
Streamlit multi-page app:
- **Home** — project overview, key stats
- **Dataset Analytics** — distributions, correlations, class balance
- **Risk Prediction** — per-student risk lookup + batch view
- **Grade Prediction** — regression results, actual-vs-predicted
- **Student Clusters** — segment visualization (PCA projection), segment profiles
- **Explainable AI** — SHAP summary + per-student waterfall, plain-English explanation
- **Recommendations** — intervention list per student/segment
- **Admin Panel** — model metrics, retraining trigger, data refresh status

### Documentation 
- Installation guide, user manual, API documentation (this README + `docs/`)
- Architecture diagram, ER diagram, flowchart (`docs/diagrams/`)

### GitHub Packaging 
- `.gitignore` , `requirements.txt` 
- Clean, phase-based commit history
- This README as the top-level project entry point

---

## Engineered Features

| Feature | OULAD (behavioral) | UCI (survey-based) |
|---|---|---|
| **Engagement Score** | Direct — total VLE clicks + active days | Proxy — `studytime` + `activities` |
| **Attendance Consistency** | Proxy — inverse coefficient of variation of weekly clicks | Direct — inverse of `absences` |
| **Assignment Timeliness** | Direct — days early/late vs. due date | Proxy (weak) — inverse of `failures` |
| **Quiz Performance Trend** | Direct — slope of assessment scores over time | Direct — slope across G1→G2→G3 |
| **LMS Activity Score** | Direct — diversity of VLE activity types used | Proxy (weakest) — `internet` access + `studytime` |
| **Study Efficiency** | Direct — score per unit of clicking effort | Direct — G3 per unit of `studytime` |
| **Academic Health Score** | Weighted composite of all above | Weighted composite (proxies downweighted) |

**Validation:** OULAD's Academic Health Score is monotonic with real outcomes — mean score rises Withdrawn (26.9) → Fail (39.6) → Pass (52.9) → Distinction (57.9) — despite never seeing the label during construction.

---

## Machine Learning Modules

### Module 1 — Academic Risk Prediction (OULAD) 
3-class classification: Low / Medium / High risk, mapped from `final_result` (Withdrawn→High, Fail→Medium, Pass/Distinction→Low).

| Model | Accuracy | F1 (macro) | ROC-AUC (OvR) |
|---|---|---|---|
| Logistic Regression | 0.738 | 0.687 | 0.881 |
| Random Forest | 0.770 | 0.718 | 0.903 |
| **XGBoost (best)** | **0.780** | **0.719** | **0.904** |

SMOTE applied inside an `imblearn.Pipeline`, fit only on the training fold, to address class imbalance (47% Low / 31% High / 22% Medium) without leaking test-set information.

### Module 2 — Final Grade Prediction (UCI) 
Regression on G3 (final grade, 0-20), deliberately **excluding G1/G2** as predictors (and excluding any Phase-3 feature derived from G3) to keep this a genuine early-warning problem rather than a near-tautological one.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Linear Regression (best)** | **2.62** | **3.71** | **0.111** |
| Random Forest | 2.63 | 3.74 | 0.097 |
| XGBoost | 2.66 | 3.74 | 0.096 |

### Module 3 — Student Segmentation (OULAD) 
K-Means (k=3, chosen via elbow + silhouette analysis across k=2-8) into **At Risk / Average / High Performer**, with cluster identity assigned post-hoc by ranking centroids on mean Academic Health Score (never fit on the label).

- Segment sizes: High Performer 17,362 · Average 11,865 · At Risk 3,366
- Validation: mean `final_result` rises monotonically across segments (0.11 → 0.88 → 1.73), confirming the clusters found real structure unsupervised.

---

## Explainable AI

Uses SHAP's `TreeExplainer` against the best-performing tree model (XGBoost) for exact, fast SHAP value computation. Produces:
- **Global summary plot** — which features matter most, per risk class
- **Local waterfall plot** — exactly how one student's prediction was built up, feature by feature
- **Force plot** — compact visual alternative to the waterfall
- **Plain-English translation** — e.g. *"This student is predicted as High Risk. Factors increasing this risk: low engagement score (12.4), low attendance consistency (34.1). Factors reducing this risk: high assignment timeliness (88.2)."*

The underlying scaler + classifier are extracted from the trained SMOTE pipeline before SHAP computation, since SHAP needs to explain the classifier's behavior on the same (scaled) input space it was trained on — SMOTE itself has no effect at inference time and is not part of the explanation.

---

## Recommendation Engine

Rule-based, per-student, generated dynamically from the engineered feature table (not hardcoded per student). Each rule fires off a specific feature crossing a documented threshold, and cites that feature back to the advisor so the recommendation is itself explainable:

```
Attendance Consistency < 40  →  "Recommend a mentor check-in — engagement has been irregular."
LMS Activity Score < 30      →  "Recommend a structured daily learning schedule."
Quiz Performance Trend < 0   →  "Recommend targeted revision sessions — recent scores are trending down."
Assignment Timeliness < 50   →  "Recommend a deadline-management conversation."
```

Multiple rules can fire for the same student; recommendations are ranked by how far the triggering feature is from its threshold (larger gap → higher priority).

---



## Database

Planned schema (SQLAlchemy models + raw SQL DDL will both be provided in `database/`):

```sql
CREATE TABLE students (
    student_key   VARCHAR PRIMARY KEY,   -- code_module|code_presentation|id_student
    dataset       VARCHAR,               -- 'oulad' or 'uci'
    features_json TEXT,                  -- engineered feature snapshot
    created_at    TIMESTAMP
);

CREATE TABLE predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    student_key   VARCHAR REFERENCES students(student_key),
    prediction_type VARCHAR,             -- 'risk' | 'grade' | 'cluster'
    prediction_value VARCHAR,
    model_name    VARCHAR,
    model_version VARCHAR,
    predicted_at  TIMESTAMP
);

CREATE TABLE shap_explanations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    student_key   VARCHAR REFERENCES students(student_key),
    feature_name  VARCHAR,
    shap_value    FLOAT,
    raw_value     FLOAT
);

CREATE TABLE recommendations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    student_key   VARCHAR REFERENCES students(student_key),
    recommendation TEXT,
    triggering_feature VARCHAR,
    priority      INTEGER,
    generated_at  TIMESTAMP
);
```

---

## Dashboard

Streamlit multi-page app (Home, Dataset Analytics, Risk Prediction, Grade Prediction, Student Clusters, Explainable AI, Recommendations, Admin Panel) — full build documented.

---

## Installation

```bash
git clone <repo-url>
cd CogniMap
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place raw data files (not committed to git) here:
```
data/raw/oulad/  # studentInfo.csv, studentRegistration.csv, studentAssessment.csv,
                 # assessments.csv, vle.csv, studentVle.csv, courses.csv
data/raw/uci/    # student-mat.csv, student-por.csv
```

## Testing

```bash
python -m pytest -m "not slow" -v      # fast unit tests
python -m pytest -m slow -v            # full integration tests against real data
python -m pytest --cov=src             # coverage report
```

## Project Structure

```
# CogniMap — Explainable Early-Warning System for Academic Risk & Performance Prediction

CogniMap is an end-to-end, production-structured machine learning platform that predicts student academic risk, final grades, and performance segments — and explains every prediction in plain English, with personalized intervention recommendations attached.

Built as a portfolio-grade demonstration of the **full ML engineering lifecycle**: data ingestion → cleaning → feature engineering → modeling → explainability → recommendation → API → dashboard, using two real, independent educational datasets.

---

## Table of Contents

- [Why This Project Exists](#why-this-project-exists)
- [Datasets](#datasets)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Feature Catalogue](#feature-catalogue-full-lifecycle)
- [Engineered Features](#engineered-features)
- [Machine Learning Modules](#machine-learning-modules)
- [Explainable AI](#explainable-ai)
- [Recommendation Engine](#recommendation-engine)
- [Backend API](#backend-api)
- [Database](#database)
- [Dashboard](#dashboard)
- [Installation](#installation)
- [Testing](#testing)
- [Project Structure](#project-structure)

---

## Why This Project Exists

Most student-performance-prediction projects on GitHub use a single small dataset (usually just the UCI Student Performance dataset) and stop at a single classifier with an accuracy score. EduRadar is deliberately broader:

- **Two real, independent datasets** (OULAD + UCI), used for the supervised tasks each one actually supports, rather than forced together.
- **A composite, validated "Academic Health Score"** that correlates monotonically with real outcomes — without ever seeing the label during construction.
- **Explicit leakage prevention**, including a caught case of engineered features that indirectly encoded their own target.
- **Explainability as a first-class feature**, not an afterthought — every prediction should be answerable with "why."
- **A recommendation layer** that turns predictions into concrete next actions for an advisor, not just a risk score.

---

## Datasets

### 1. Open University Learning Analytics Dataset (OULAD)
- 32,593 students across 7 relational tables, including a real VLE (Virtual Learning Environment) clickstream log (~10.6M interaction events).
- Provides genuine **behavioral** data: demographics, assessment scores, day-by-day engagement, and final outcome (`Withdrawn` / `Fail` / `Pass` / `Distinction`).
- Used for: **Risk Classification (Module 1)** and **Segmentation (Module 3)**, since it has real dropout/outcome labels and rich behavioral signal.

### 2. UCI Student Performance Dataset
- 1,044 records (395 Math + 649 Portuguese), with rich demographic/socio-economic survey data and three real grade checkpoints (G1, G2, G3, each 0-20).
- Used for: **Grade Regression (Module 2)**, since it has a genuine continuous target that OULAD lacks.

**Design principle:** these are treated as two separate pipelines feeding a common feature schema, not naively merged — different student populations, granularities, and targets mean row-level concatenation would create fake associations. Where a feature can't be measured the same way in both datasets (e.g. UCI has no clickstream), this is documented as a **proxy**, not silently treated as equivalent.

---

## Architecture

```
Raw Data (OULAD + UCI)
        │
        ▼
┌───────────────────┐
│  Preprocessing     │  cleaning, dedup, encoding (src/preprocessing/)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Feature Engineering│  7 core features, per-dataset (src/feature_engineering/)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   Machine Learning  │  risk / grade / segmentation models (src/models/)
└───────────────────┘
        │
        ▼
┌───────────────────┐        ┌───────────────────┐
│  Explainability     │──────▶│ Recommendation Eng. │
│  (src/explainability)│      │ (src/recommendation)│
└───────────────────┘        └───────────────────┘
        │                              │
        └──────────────┬───────────────┘
                        ▼
              ┌───────────────────┐
              │   FastAPI Backend  │  (api/)
              └───────────────────┘
                        │
        ┌───────────────┴───────────────┐
        ▼                                ▼
┌───────────────┐              ┌───────────────────┐
│  Database       │              │ Streamlit Dashboard│
│ (SQLite/Postgres)│◀────────────│    (dashboard/)     │
└───────────────┘              └───────────────────┘
```

`api/` and `dashboard/` are thin presentation layers — they import logic from `src/`, never reimplement it, so predictions/explanations/recommendations are computed identically everywhere they're used.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Backend | FastAPI, Pydantic, Uvicorn |
| Database | SQLAlchemy (SQLite dev / PostgreSQL prod) |
| Dashboard | Streamlit, Plotly |
| Testing | pytest, pytest-cov, httpx |
| Utilities | loguru (logging), joblib (model persistence) |

---

## Feature Catalogue (Full Lifecycle)

### Data Processing 
- Missing value handling (documented per-column strategy — never blanket mean/mode imputation)
- Duplicate removal (caught 787,170 duplicate clickstream rows in OULAD)
- Categorical encoding (ordinal domain-knowledge maps + one-hot for nominal fields)
- Class imbalance handling via SMOTE (applied leakage-safely, inside a pipeline fit only on training folds)
- Reusable, testable preprocessing pipelines (`BaseCleaner` abstract class, Template Method pattern)

### Feature Engineering 
7 named features built per-dataset (see [Engineered Features](#engineered-features) below), each with an explicit "direct measurement" or "proxy" classification.

### Machine Learning 
Three supervised/unsupervised modules (see [Machine Learning Modules](#machine-learning-modules) below), each using the dataset whose real target fits the task.

### Explainable AI 
- Global feature importance (SHAP summary plot, per risk class)
- Local, per-student explanation (top contributing features, signed direction)
- Waterfall plot per student
- Force plot per student
- Plain-English translation of SHAP output for non-technical advisors

### Recommendation Engine 
Rule-based, dynamically generated interventions mapped from feature thresholds, e.g.:
- Low attendance consistency → recommend mentor meeting
- Low LMS/engagement activity → recommend a daily structured learning schedule
- Declining quiz performance trend → recommend targeted revision sessions
- Poor assignment timeliness → recommend a deadline-management check-in

Recommendations will be generated per-student from the same feature table used for prediction, and will cite which feature(s) triggered which recommendation.

### Backend API 
FastAPI endpoints (planned contracts, finalized during Phase 7):

| Method | Endpoint | Description |
|---|---|---|
| GET | `/student/{id}` | Full profile: features, current predictions |
| POST | `/predict/risk` | Risk classification (Low/Medium/High) |
| POST | `/predict/grade` | Final grade regression |
| POST | `/predict/cluster` | Segment assignment (At Risk/Average/High Performer) |
| GET | `/explain/{id}` | SHAP explanation, plain-English + structured |
| GET | `/recommendation/{id}` | Personalized intervention list |

### Database 
Planned schema (SQLite for local dev, PostgreSQL-ready via SQLAlchemy):
- `students` — demographic + engineered feature snapshot
- `predictions` — risk/grade/cluster predictions with timestamps and model version
- `shap_explanations` — stored per-student SHAP contributions
- `recommendations` — generated interventions, linked to the feature(s) that triggered them

### Dashboard 
Streamlit multi-page app:
- **Home** — project overview, key stats
- **Dataset Analytics** — distributions, correlations, class balance
- **Risk Prediction** — per-student risk lookup + batch view
- **Grade Prediction** — regression results, actual-vs-predicted
- **Student Clusters** — segment visualization (PCA projection), segment profiles
- **Explainable AI** — SHAP summary + per-student waterfall, plain-English explanation
- **Recommendations** — intervention list per student/segment
- **Admin Panel** — model metrics, retraining trigger, data refresh status

### Documentation 
- Installation guide, user manual, API documentation (this README + `docs/`)
- Architecture diagram, ER diagram, flowchart (`docs/diagrams/`)

### GitHub Packaging 
- `.gitignore` , `requirements.txt` 
- Clean, phase-based commit history
- This README as the top-level project entry point

---

## Engineered Features

| Feature | OULAD (behavioral) | UCI (survey-based) |
|---|---|---|
| **Engagement Score** | Direct — total VLE clicks + active days | Proxy — `studytime` + `activities` |
| **Attendance Consistency** | Proxy — inverse coefficient of variation of weekly clicks | Direct — inverse of `absences` |
| **Assignment Timeliness** | Direct — days early/late vs. due date | Proxy (weak) — inverse of `failures` |
| **Quiz Performance Trend** | Direct — slope of assessment scores over time | Direct — slope across G1→G2→G3 |
| **LMS Activity Score** | Direct — diversity of VLE activity types used | Proxy (weakest) — `internet` access + `studytime` |
| **Study Efficiency** | Direct — score per unit of clicking effort | Direct — G3 per unit of `studytime` |
| **Academic Health Score** | Weighted composite of all above | Weighted composite (proxies downweighted) |

**Validation:** OULAD's Academic Health Score is monotonic with real outcomes — mean score rises Withdrawn (26.9) → Fail (39.6) → Pass (52.9) → Distinction (57.9) — despite never seeing the label during construction.

---

## Machine Learning Modules

### Module 1 — Academic Risk Prediction (OULAD) 
3-class classification: Low / Medium / High risk, mapped from `final_result` (Withdrawn→High, Fail→Medium, Pass/Distinction→Low).

| Model | Accuracy | F1 (macro) | ROC-AUC (OvR) |
|---|---|---|---|
| Logistic Regression | 0.738 | 0.687 | 0.881 |
| Random Forest | 0.770 | 0.718 | 0.903 |
| **XGBoost (best)** | **0.780** | **0.719** | **0.904** |

SMOTE applied inside an `imblearn.Pipeline`, fit only on the training fold, to address class imbalance (47% Low / 31% High / 22% Medium) without leaking test-set information.

### Module 2 — Final Grade Prediction (UCI) 
Regression on G3 (final grade, 0-20), deliberately **excluding G1/G2** as predictors (and excluding any Phase-3 feature derived from G3) to keep this a genuine early-warning problem rather than a near-tautological one.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Linear Regression (best)** | **2.62** | **3.71** | **0.111** |
| Random Forest | 2.63 | 3.74 | 0.097 |
| XGBoost | 2.66 | 3.74 | 0.096 |

### Module 3 — Student Segmentation (OULAD) 
K-Means (k=3, chosen via elbow + silhouette analysis across k=2-8) into **At Risk / Average / High Performer**, with cluster identity assigned post-hoc by ranking centroids on mean Academic Health Score (never fit on the label).

- Segment sizes: High Performer 17,362 · Average 11,865 · At Risk 3,366
- Validation: mean `final_result` rises monotonically across segments (0.11 → 0.88 → 1.73), confirming the clusters found real structure unsupervised.

---

## Explainable AI

Uses SHAP's `TreeExplainer` against the best-performing tree model (XGBoost) for exact, fast SHAP value computation. Produces:
- **Global summary plot** — which features matter most, per risk class
- **Local waterfall plot** — exactly how one student's prediction was built up, feature by feature
- **Force plot** — compact visual alternative to the waterfall
- **Plain-English translation** — e.g. *"This student is predicted as High Risk. Factors increasing this risk: low engagement score (12.4), low attendance consistency (34.1). Factors reducing this risk: high assignment timeliness (88.2)."*

The underlying scaler + classifier are extracted from the trained SMOTE pipeline before SHAP computation, since SHAP needs to explain the classifier's behavior on the same (scaled) input space it was trained on — SMOTE itself has no effect at inference time and is not part of the explanation.

---

## Recommendation Engine

Rule-based, per-student, generated dynamically from the engineered feature table (not hardcoded per student). Each rule fires off a specific feature crossing a documented threshold, and cites that feature back to the advisor so the recommendation is itself explainable:

```
Attendance Consistency < 40  →  "Recommend a mentor check-in — engagement has been irregular."
LMS Activity Score < 30      →  "Recommend a structured daily learning schedule."
Quiz Performance Trend < 0   →  "Recommend targeted revision sessions — recent scores are trending down."
Assignment Timeliness < 50   →  "Recommend a deadline-management conversation."
```

Multiple rules can fire for the same student; recommendations are ranked by how far the triggering feature is from its threshold (larger gap → higher priority).

---



## Database

Planned schema (SQLAlchemy models + raw SQL DDL will both be provided in `database/`):

```sql
CREATE TABLE students (
    student_key   VARCHAR PRIMARY KEY,   -- code_module|code_presentation|id_student
    dataset       VARCHAR,               -- 'oulad' or 'uci'
    features_json TEXT,                  -- engineered feature snapshot
    created_at    TIMESTAMP
);

CREATE TABLE predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    student_key   VARCHAR REFERENCES students(student_key),
    prediction_type VARCHAR,             -- 'risk' | 'grade' | 'cluster'
    prediction_value VARCHAR,
    model_name    VARCHAR,
    model_version VARCHAR,
    predicted_at  TIMESTAMP
);

CREATE TABLE shap_explanations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    student_key   VARCHAR REFERENCES students(student_key),
    feature_name  VARCHAR,
    shap_value    FLOAT,
    raw_value     FLOAT
);

CREATE TABLE recommendations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    student_key   VARCHAR REFERENCES students(student_key),
    recommendation TEXT,
    triggering_feature VARCHAR,
    priority      INTEGER,
    generated_at  TIMESTAMP
);
```

---

## Dashboard

Streamlit multi-page app (Home, Dataset Analytics, Risk Prediction, Grade Prediction, Student Clusters, Explainable AI, Recommendations, Admin Panel) — full build documented.

---

## Installation

```bash
git clone <repo-url>
cd CogniMap
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place raw data files (not committed to git) here:
```
data/raw/oulad/  # studentInfo.csv, studentRegistration.csv, studentAssessment.csv,
                 # assessments.csv, vle.csv, studentVle.csv, courses.csv
data/raw/uci/    # student-mat.csv, student-por.csv
```

## Testing

```bash
python -m pytest -m "not slow" -v      # fast unit tests
python -m pytest -m slow -v            # full integration tests against real data
python -m pytest --cov=src             # coverage report
```

## Project Structure

```
CogniMap/
├── data/
│   ├── raw/{oulad,uci}/          # untouched source files (git-ignored)
│   ├── processed/{oulad,uci,features}/   # cleaned + engineered parquet files
│   └── external/
├── notebooks/                    # exploration only — no production logic
├── src/
│   ├── preprocessing/           
│   ├── feature_engineering/      
│   ├── models/                   
│   ├── explainability/           
│   ├── recommendation/          
│   └── utils/                    # config, logging
├── api/                         
├── dashboard/                   
├── database/                   
├── tests/                        # mirrors src/ structure
├── models/                       # serialized model artifacts (git-ignored)
├── docs/                         # diagrams, manuals, plots
├── requirements.txt
├── .gitignore
└── main.py
```
/
├── data/
│   ├── raw/{oulad,uci}/          # untouched source files (git-ignored)
│   ├── processed/{oulad,uci,features}/   # cleaned + engineered parquet files
│   └── external/
├── notebooks/                    # exploration only — no production logic
├── src/
│   ├── preprocessing/           
│   ├── feature_engineering/      
│   ├── models/                   
│   ├── explainability/           
│   ├── recommendation/          
│   └── utils/                    # config, logging
├── api/                         
├── dashboard/                   
├── database/                   
├── tests/                        # mirrors src/ structure
├── models/                       # serialized model artifacts (git-ignored)
├── docs/                         # diagrams, manuals, plots
├── requirements.txt
├── .gitignore
└── main.py
```
/
├── data/
│   ├── raw/{oulad,uci}/          # untouched source files (git-ignored)
│   ├── processed/{oulad,uci,features}/   # cleaned + engineered parquet files
│   └── external/
├── notebooks/                    # exploration only — no production logic
├── src/
│   ├── preprocessing/           
│   ├── feature_engineering/      
│   ├── models/                   
│   ├── explainability/           
│   ├── recommendation/          
│   └── utils/                    # config, logging
├── api/                         
├── dashboard/                   
├── database/                   
├── tests/                        # mirrors src/ structure
├── models/                       # serialized model artifacts (git-ignored)
├── docs/                         # diagrams, manuals, plots
├── requirements.txt
├── .gitignore
└── main.py
```
