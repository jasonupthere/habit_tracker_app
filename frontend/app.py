from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


st.set_page_config(page_title="AI Habit Tracker", layout="wide")


def _get_backend_url() -> str:
    try:
        backend_from_secrets = st.secrets.get("BACKEND_URL", None)
    except StreamlitSecretNotFoundError:
        backend_from_secrets = None

    return backend_from_secrets or st.sidebar.text_input(
        "Backend URL",
        value="http://127.0.0.1:8001",
        help="Where FastAPI is running (default: local).",
    )


BACKEND_URL = _get_backend_url()


def _get(path: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
    return requests.get(f"{BACKEND_URL}{path}", params=params, timeout=30)


def _post(path: str, payload: Dict[str, Any]) -> requests.Response:
    return requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=60)


def _safe_json(resp: requests.Response) -> Dict[str, Any]:
    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text}


st.title("AI-Powered Habit Tracker (Failure Prediction + A* Recovery Plan)")

with st.sidebar:
    st.subheader("Connection")
    if st.button("Check Backend Health", use_container_width=True):
        try:
            r = _get("/health")
            if r.status_code == 200:
                st.success("Backend OK")
            else:
                st.error(f"Backend error: {r.status_code} {r.text}")
        except Exception as e:
            st.error(f"Cannot reach backend: {e}")


st.subheader("1) Habits")

col_add, col_list = st.columns([1, 2])

with col_add:
    new_name = st.text_input("New habit name", placeholder="Gym, Study, Read, ...")
    if st.button("Add Habit", type="primary", use_container_width=True, disabled=not new_name.strip()):
        try:
            r = _post("/add_habit", {"name": new_name.strip()})
            if r.status_code == 200:
                st.success("Habit added.")
                st.session_state["selected_habit_id"] = _safe_json(r)["habit_id"]
            else:
                st.error(f"{r.status_code}: {_safe_json(r)}")
        except Exception as e:
            st.error(f"Request failed: {e}")

with col_list:
    habits: List[Dict[str, Any]] = []
    try:
        r = _get("/habits")
        if r.status_code == 200:
            habits = r.json()
    except Exception:
        habits = []

    if not habits:
        st.info("No habits yet. Add one to start tracking.")
        st.stop()

    habit_labels = {h["habit_id"]: h["name"] for h in habits}
    default_id = st.session_state.get("selected_habit_id", habits[0]["habit_id"])
    if default_id not in habit_labels:
        default_id = habits[0]["habit_id"]

    selected_habit_id = st.selectbox(
        "Select habit",
        options=list(habit_labels.keys()),
        format_func=lambda hid: f"{habit_labels[hid]} (id={hid})",
        index=list(habit_labels.keys()).index(default_id),
    )
    st.session_state["selected_habit_id"] = selected_habit_id


st.subheader("2) Daily Logging")

log_col1, log_col2, log_col3, log_col4 = st.columns([1, 1, 1, 1])

with log_col1:
    log_date = st.date_input("Log date", value=date.today())
with log_col2:
    if st.button("✅ Done (1)", use_container_width=True):
        r = _post(
            "/log",
            {"habit_id": int(selected_habit_id), "date": log_date.isoformat(), "status": 1},
        )
        if r.status_code == 200:
            st.success("Logged: Done")
        else:
            st.error(f"{r.status_code}: {_safe_json(r)}")
with log_col3:
    if st.button("❌ Missed (0)", use_container_width=True):
        r = _post(
            "/log",
            {"habit_id": int(selected_habit_id), "date": log_date.isoformat(), "status": 0},
        )
        if r.status_code == 200:
            st.success("Logged: Missed")
        else:
            st.error(f"{r.status_code}: {_safe_json(r)}")
with log_col4:
    if st.button("Refresh", use_container_width=True):
        st.rerun()


info_left, info_right = st.columns([1, 2])

with info_left:
    st.subheader("Stats")
    try:
        r = _get("/habit_info", params={"habit_id": int(selected_habit_id)})
        if r.status_code == 200:
            info = r.json()
            st.metric("Current streak", int(info["streak"]))
            st.metric("Consistency (last 30d)", f"{float(info['consistency'])*100:.1f}%")
        else:
            st.error(f"{r.status_code}: {_safe_json(r)}")
    except Exception as e:
        st.error(f"Cannot load stats: {e}")

with info_right:
    st.subheader("History")
    try:
        r = _get("/history", params={"habit_id": int(selected_habit_id)})
        if r.status_code == 200:
            hist = r.json()
            df = pd.DataFrame({"date": hist["dates"], "status": hist["status"]})
            if not df.empty:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=df["date"],
                        y=df["status"],
                        mode="lines+markers",
                        name="Done (1) / Missed (0)",
                    )
                )
                fig.update_layout(
                    height=320,
                    yaxis=dict(title="Status", tickmode="array", tickvals=[0, 1]),
                    xaxis_title="Date",
                    title="Habit log timeline",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No logs yet for this habit.")
        else:
            st.error(f"{r.status_code}: {_safe_json(r)}")
    except Exception as e:
        st.error(f"Cannot load history: {e}")


st.divider()
st.subheader("3) Failure Prediction (ML)")

pred_col1, pred_col2, pred_col3 = st.columns([1, 1, 1])
with pred_col1:
    if st.button("Train Model", type="primary", use_container_width=True):
        try:
            r = _post("/train", {"habit_id": int(selected_habit_id)})
            if r.status_code == 200:
                tr = r.json()
                st.success("Model trained.")
                st.json(tr)
            else:
                st.error(f"{r.status_code}: {_safe_json(r)}")
        except Exception as e:
            st.error(f"Training request failed: {e}")

with pred_col2:
    predict_for = st.date_input("Predict failure for date", value=date.today() + timedelta(days=1))

with pred_col3:
    if st.button("Predict Failure", use_container_width=True):
        try:
            r = _post(
                "/predict",
                {"habit_id": int(selected_habit_id), "date": predict_for.isoformat()},
            )
            if r.status_code == 200:
                out = r.json()
                p = float(out["failure_probability"])
                st.metric("Failure probability", f"{p*100:.1f}%")
                st.write(f"Risk level: **{out['risk_level']}**")
            else:
                st.error(f"{r.status_code}: {_safe_json(r)}")
        except Exception as e:
            st.error(f"Predict request failed: {e}")


st.divider()
st.subheader("4) A* Habit Recovery Plan")

plan_col1, plan_col2, plan_col3 = st.columns([1, 1, 1])
with plan_col1:
    plan_start = st.date_input("Plan start date", value=date.today() + timedelta(days=1))
with plan_col2:
    plan_days = st.slider("Max days", min_value=3, max_value=21, value=10, step=1)
with plan_col3:
    if st.button("Generate Recovery Plan", use_container_width=True):
        try:
            r = _post(
                "/recovery_plan",
                {
                    "habit_id": int(selected_habit_id),
                    "start_date": plan_start.isoformat(),
                    "max_days": int(plan_days),
                },
            )
            if r.status_code == 200:
                out = r.json()
                plan_df = pd.DataFrame(out["plan"])
                st.success("Recovery plan generated.")
                st.dataframe(plan_df, use_container_width=True)
            else:
                st.error(f"{r.status_code}: {_safe_json(r)}")
        except Exception as e:
            st.error(f"Plan request failed: {e}")
