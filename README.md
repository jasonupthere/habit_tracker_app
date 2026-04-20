# AI Habit Tracker Web App (FastAPI + Streamlit + ML + A*)

This is a full-stack Habit Tracker that:

1. Tracks daily habit completion (Done=1, Missed=0)
2. Trains a **failure prediction** model (Logistic Regression) with scikit-learn
3. Uses **A\*** search to generate an optimal “Habit Recovery Plan”

## Run

From:

`C:\Users\jonat\Documents\Codex\2026-04-19-build-a-full-stack-ai-powered-2`

Install deps:

```bash
py -m pip install -r requirements.txt
```

Start backend:

```bash
py -m uvicorn backend.main:app --reload --port 8001
```

Start frontend:

```bash
py -m streamlit run frontend/app.py --server.port 8501
```

Or (Windows PowerShell):

```powershell
.\scripts\run_all.ps1
```

## URLs

- Streamlit website: `http://127.0.0.1:8501`
- FastAPI docs: `http://127.0.0.1:8001/docs`

## Note

The previous Sales Forecasting app was preserved at:

`C:\Users\jonat\Documents\Codex\2026-04-19-build-a-full-stack-ai-powered-2\sales_forecasting`
