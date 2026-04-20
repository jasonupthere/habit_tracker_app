from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .astar import State, astar_recovery_plan
from .db import connect, get_habit, get_habit_by_name, init_db, parse_iso_date
from .model import (
    compute_streak_and_consistency,
    fetch_habit_logs_df,
    predict_failure_probability,
    risk_level,
    train_failure_model,
)


app = FastAPI(title="AI Habit Tracker API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AddHabitRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)


class AddHabitResponse(BaseModel):
    habit_id: int
    name: str


class LogRequest(BaseModel):
    habit_id: int
    date: str = Field(..., description="YYYY-MM-DD")
    status: int = Field(..., ge=0, le=1, description="Done=1, Missed=0")


class SimpleOK(BaseModel):
    ok: bool = True


class HabitInfo(BaseModel):
    habit_id: int
    name: str
    streak: int
    consistency: float


class TrainRequest(BaseModel):
    habit_id: int


class TrainResponse(BaseModel):
    accuracy: float
    roc_auc: Optional[float] = None
    n_samples: int
    n_positive: int


class PredictRequest(BaseModel):
    habit_id: int
    date: str = Field(..., description="Predict failure probability for this date (YYYY-MM-DD).")


class PredictResponse(BaseModel):
    habit_id: int
    date: str
    failure_probability: float
    risk_level: str


class RecoveryPlanRequest(BaseModel):
    habit_id: int
    start_date: Optional[str] = Field(None, description="YYYY-MM-DD; default is tomorrow.")
    max_days: int = Field(10, ge=3, le=21)


class RecoveryPlanResponse(BaseModel):
    habit_id: int
    start_date: str
    plan: List[Dict[str, str]]


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/habits", response_model=List[AddHabitResponse])
def list_habits() -> List[AddHabitResponse]:
    with connect() as conn:
        rows = conn.execute("SELECT id, name FROM habits ORDER BY name ASC").fetchall()
    return [AddHabitResponse(habit_id=int(r["id"]), name=str(r["name"])) for r in rows]


@app.post("/add_habit", response_model=AddHabitResponse)
def add_habit(req: AddHabitRequest) -> AddHabitResponse:
    init_db()
    existing = get_habit_by_name(req.name.strip())
    if existing:
        return AddHabitResponse(habit_id=existing[0], name=existing[1])

    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO habits(name, created_at) VALUES(?, ?)",
            (req.name.strip(), datetime.utcnow().isoformat() + "Z"),
        )
        habit_id = int(cur.lastrowid)
    return AddHabitResponse(habit_id=habit_id, name=req.name.strip())


@app.post("/log", response_model=SimpleOK)
def log_day(req: LogRequest) -> SimpleOK:
    habit = get_habit(req.habit_id)
    if not habit:
        raise HTTPException(status_code=404, detail="Habit not found.")

    try:
        d = parse_iso_date(req.date)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.") from e

    with connect() as conn:
        conn.execute(
            """
            INSERT INTO habit_logs(habit_id, log_date, status)
            VALUES(?, ?, ?)
            ON CONFLICT(habit_id, log_date) DO UPDATE SET status=excluded.status
            """,
            (req.habit_id, d.isoformat(), int(req.status)),
        )
    return SimpleOK(ok=True)


@app.get("/habit_info", response_model=HabitInfo)
def habit_info(habit_id: int) -> HabitInfo:
    habit = get_habit(habit_id)
    if not habit:
        raise HTTPException(status_code=404, detail="Habit not found.")
    df = fetch_habit_logs_df(habit_id)
    streak, consistency = compute_streak_and_consistency(df)
    return HabitInfo(habit_id=habit_id, name=habit[1], streak=streak, consistency=consistency)


@app.get("/history")
def history(habit_id: int) -> Dict[str, Any]:
    habit = get_habit(habit_id)
    if not habit:
        raise HTTPException(status_code=404, detail="Habit not found.")
    df = fetch_habit_logs_df(habit_id)
    return {
        "habit_id": habit_id,
        "dates": [d.strftime("%Y-%m-%d") for d in df["date"].tolist()],
        "status": [int(x) for x in df["status"].tolist()],
    }


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest) -> TrainResponse:
    habit = get_habit(req.habit_id)
    if not habit:
        raise HTTPException(status_code=404, detail="Habit not found.")
    try:
        m = train_failure_model(req.habit_id)
        return TrainResponse(
            accuracy=m.accuracy,
            roc_auc=m.roc_auc,
            n_samples=m.n_samples,
            n_positive=m.n_positive,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}") from e


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    habit = get_habit(req.habit_id)
    if not habit:
        raise HTTPException(status_code=404, detail="Habit not found.")
    try:
        d = parse_iso_date(req.date)
        p = predict_failure_probability(req.habit_id, d)
        return PredictResponse(
            habit_id=req.habit_id,
            date=req.date,
            failure_probability=p,
            risk_level=risk_level(p),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e


@app.post("/recovery_plan", response_model=RecoveryPlanResponse)
def recovery_plan(req: RecoveryPlanRequest) -> RecoveryPlanResponse:
    habit = get_habit(req.habit_id)
    if not habit:
        raise HTTPException(status_code=404, detail="Habit not found.")

    df = fetch_habit_logs_df(req.habit_id)
    streak, _consistency = compute_streak_and_consistency(df)

    # Recent failures proxy: misses in last 3 days.
    recent_failures = 0
    if not df.empty:
        recent_failures = int(df.tail(3)["status"].eq(0).sum())

    # Fatigue is simulated from recent intensity: more consecutive done => more fatigue,
    # misses reduce fatigue. This is a simple proxy.
    fatigue = min(10, max(0, int(streak * 1.2) - recent_failures * 2))

    start = date.today() + timedelta(days=1)
    if req.start_date:
        try:
            start = parse_iso_date(req.start_date)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid start_date. Use YYYY-MM-DD.") from e

    # Try to use ML probability if trained; otherwise fall back.
    prob: Optional[float]
    try:
        prob = predict_failure_probability(req.habit_id, start)
    except Exception:
        prob = None

    plan = astar_recovery_plan(
        start_state=State(streak=streak, fatigue=fatigue, recent_failures=recent_failures),
        start_date=start,
        prob_from_model=prob,
        max_steps=req.max_days,
    )
    return RecoveryPlanResponse(habit_id=req.habit_id, start_date=start.isoformat(), plan=plan)

