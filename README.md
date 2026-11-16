# Student Dropout Prediction

Midterm project for [ML Zoomcamp 2026](https://github.com/DataTalksClub/machine-learning-zoomcamp) (DataTalks.Club).

## Problem description

Student dropout is one of the biggest problems higher education institutions face: every student who leaves represents lost tuition, wasted public funding and, above all, a personal setback for the student. Most dropouts are preceded by warning signs — failed courses, unpaid tuition, low grades — but by the time someone notices, it is usually too late to help.

This project builds a model that **predicts whether a student will drop out** using information available at enrollment (demographics, socio-economic background, previous education) and their academic performance during the first two semesters.

**How the solution would be used:** at the end of the first year, a student-support office sends each student's record to the prediction API and receives a dropout probability. Students with a high probability are flagged for early intervention (tutoring, financial aid counselling, mentoring) before they actually leave.

The original dataset has three classes (`Dropout`, `Enrolled`, `Graduate`). Since the goal is to detect students at risk, the problem is framed as **binary classification**: `dropout = 1` if the student dropped out, `0` otherwise. The metric is **ROC AUC**, because the classes are imbalanced (32% dropouts) and the support office can choose its own probability threshold depending on how many students it can follow up.

## Dataset

[Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697) — UCI Machine Learning Repository (CC BY 4.0).
4,424 students from a Portuguese polytechnic institute, 36 features, no missing values.

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success*. https://doi.org/10.24432/C5MC89

The dataset is committed in `data/data.csv`. To download it again (PowerShell):

```powershell
Invoke-WebRequest "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip" -OutFile data.zip
Expand-Archive data.zip -DestinationPath data
```

## Project structure

| File | Purpose |
|---|---|
| `notebook.ipynb` | Data preparation, EDA, feature importance, model selection and tuning |
| `train.py` | Trains the final model and saves it to `model.bin` |
| `predict.py` | FastAPI web service that loads the model and serves predictions |
| `test.py` | Sends a sample student to the running service |
| `pyproject.toml`, `uv.lock` | Dependencies (managed with [uv](https://docs.astral.sh/uv/)) |
| `Dockerfile` | Container image for the web service |

## Results

| Model | Validation AUC |
|---|---|
| **XGBoost** (`max_depth=4`, `learning_rate=0.05`, `n_estimators=200`) | **0.922** |
| Logistic regression (`C=0.1`) | 0.921 |
| Random forest (`max_depth=15`, `n_estimators=50`) | 0.918 |
| Decision tree (`max_depth=6`, `min_samples_leaf=50`) | 0.904 |

The final XGBoost model, retrained on train + validation, reaches **AUC 0.916 on the test set**.

The most important features are the number of approved courses and grades in both semesters and whether tuition fees are up to date. Macro-economic variables (unemployment, inflation, GDP) add very little.

## How to run

### 1. Install dependencies

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/). uv downloads Python 3.11 automatically.

```bash
uv sync
```

### 2. Train the model

```bash
uv run python train.py
```

This creates `model.bin`.

### 3. Run the web service locally

```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
```

In another terminal:

```bash
uv run python test.py
```

Expected output:

```
{'dropout_probability': 0.1225, 'dropout': False}
```

You can also open http://localhost:9696/docs, click **POST /predict → Try it out → Execute** and test it with the example student included there (a student who dropped out; the model returns a probability of about 0.95).

### 4. Run with Docker

```bash
docker build -t student-dropout:v1 .
docker run --rm -p 9696:9696 student-dropout:v1
```

Then run `uv run python test.py` again, or use http://localhost:9696/docs.

## API

`POST /predict` — receives one student as JSON (36 fields, same names as the cleaned dataset columns) and returns:

```json
{"dropout_probability": 0.954, "dropout": true}
```

`GET /health` — returns `{"status": "ok"}`.

## Deployment

The service runs in Docker locally. Screenshot of the service responding:

![Service running in Docker](images/docker_predict.png)
