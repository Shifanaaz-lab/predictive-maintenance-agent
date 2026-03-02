# dashboard.py

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pymongo import MongoClient
from datetime import datetime
import io
import numpy as np

st.set_page_config(layout="wide")

st.title("✈️ Industrial Predictive Maintenance Platform")

# ==============================
# Load Data & Model
# ==============================

df = pd.read_csv("data/engineered_train.csv")
model = joblib.load("models/model.pkl")

engine_ids = df["engine_id"].unique()

# MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["predictive_maintenance"]
decision_collection = db["decision_logs"]

tab1, tab2 = st.tabs(["📊 Model Analytics", "🏭 Operations View"])

# ==========================================================
# ===================== TAB 1 ==============================
# ==========================================================

with tab1:

    st.subheader("Model Performance Overview")

    col1, col2 = st.columns(2)
    col1.metric("Cross-Validated R²", "0.926 ± 0.0076")
    col2.metric("Mean Absolute Error", "~7.5 cycles")

    st.markdown("---")

    st.subheader("Feature Importance")

    importances = model.feature_importances_
    feature_names = df.drop(columns=["engine_id", "cycle", "RUL"]).columns

    importance_df = (
        pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        })
        .sort_values(by="importance", ascending=False)
        .head(20)
    )

    fig_imp = px.bar(
        importance_df,
        x="importance",
        y="feature",
        orientation="h",
        title="Top 20 Features Driving Degradation"
    )

    st.plotly_chart(fig_imp, use_container_width=True)


# ==========================================================
# ===================== TAB 2 ==============================
# ==========================================================

with tab2:

    st.subheader("Fleet Health Summary (AI Predicted)")

    simulated_rows = []

    for engine_id, group in df.groupby("engine_id"):
        random_index = np.random.randint(0, len(group))
        simulated_rows.append(group.iloc[random_index])

    fleet_df = pd.DataFrame(simulated_rows)

    X_fleet = fleet_df.drop(columns=["engine_id", "cycle", "RUL"])
    predicted_rul = model.predict(X_fleet)

    fleet_df["predicted_RUL"] = predicted_rul
    fleet_df["failure_probability"] = 1 / (1 + np.exp((predicted_rul - 50)/10))

    critical_count = (predicted_rul < 30).sum()
    warning_count = ((predicted_rul >= 30) & (predicted_rul < 80)).sum()
    normal_count = (predicted_rul >= 80).sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 Critical Engines", int(critical_count))
    col2.metric("🟠 Warning Engines", int(warning_count))
    col3.metric("🟢 Normal Engines", int(normal_count))

    st.markdown("---")

    # =========================
    # Engine Drill-Down
    # =========================

    st.subheader("Engine Investigation")

    selected_engine = st.selectbox("Select Engine", engine_ids)

    engine_df = df[df["engine_id"] == selected_engine]

    selected_cycle = st.slider(
        "Select Operational Cycle",
        int(engine_df["cycle"].min()),
        int(engine_df["cycle"].max()),
        int(engine_df["cycle"].max())
    )

    current_row = engine_df[engine_df["cycle"] == selected_cycle]
    X_current = current_row.drop(columns=["engine_id", "cycle", "RUL"])

    current_rul = float(model.predict(X_current)[0])
    life_ratio = float(current_row["life_ratio"].values[0])
    failure_prob = 1 / (1 + np.exp((current_rul - 50)/10))

    colA, colB, colC = st.columns(3)

    colA.metric("Current Cycle", selected_cycle)
    colB.metric("Predicted Remaining Life", f"{round(current_rul,2)} cycles")
    colC.metric("Failure Probability", f"{round(failure_prob*100,2)} %")

    st.markdown("---")

    # =========================
    # Decision Logic
    # =========================

    if current_rul < 30:
        risk = "CRITICAL"
        action = "Immediate maintenance required."
        icon = "🔴"
    elif current_rul < 80:
        risk = "WARNING"
        action = "Schedule preventive maintenance."
        icon = "🟠"
    else:
        risk = "NORMAL"
        action = "Continue monitoring."
        icon = "🟢"

    st.markdown(f"## {icon} {risk}")
    st.write("**Recommended Action:**")
    st.code(action)

    st.markdown("---")

    fig_rul = px.line(
        engine_df,
        x="cycle",
        y="RUL",
        title="Historical Degradation Curve"
    )

    st.plotly_chart(fig_rul, use_container_width=True)

    st.markdown("---")

    # =========================
    # AI Explanation
    # =========================

    st.subheader("🧠 AI Decision Rationale")

    explanation = f"""
    The AI model predicts approximately {round(current_rul,2)} cycles remaining.

    Estimated probability of failure is {round(failure_prob*100,2)}%.

    Lifecycle progression is {round(life_ratio*100,2)}%.

    The engine is classified as '{risk}' based on learned degradation dynamics.
    """

    st.info(explanation)

    # =========================
    # Mongo Logging
    # =========================

       # =========================
    # Mongo Logging (Standardized Schema)
    # =========================

    if st.button("Log Decision to Database"):

        # Define priority explicitly
        if current_rul < 30:
            priority = "CRITICAL"
        elif current_rul < 80:
            priority = "HIGH"
        else:
            priority = "LOW"

        decision_collection.insert_one({
            "engine_id": int(selected_engine),
            "cycle": int(selected_cycle),
            "predicted_RUL": float(current_rul),
            "failure_probability": float(failure_prob),
            "risk_level": risk,
            "priority": priority,
            "recommended_action": action,
            "model_version": "v1.0",
            "timestamp": datetime.utcnow()
        })

        st.success("Decision logged successfully.")    # =========================
    # Show Recent Logs (Method 4)
    # =========================

    st.markdown("---")
    st.subheader("📜 Recent Logged Decisions")

    recent_logs = list(
        decision_collection.find()
        .sort("timestamp", -1)
        .limit(5)
    )

    if recent_logs:
        logs_df = pd.DataFrame(recent_logs)
        logs_df.drop(columns=["_id"], inplace=True, errors="ignore")
        st.dataframe(logs_df)
    else:
        st.write("No decisions logged yet.")

    # =========================
    # Report Export
    # =========================

    if st.button("Generate Maintenance Report"):

        report = f"""
INDUSTRIAL MAINTENANCE REPORT
----------------------------------------

Engine ID: {selected_engine}
Cycle: {selected_cycle}

Predicted RUL: {round(current_rul,2)} cycles
Failure Probability: {round(failure_prob*100,2)} %
Risk Level: {risk}

Recommended Action:
{action}

Generated at: {datetime.now()}
"""

        buffer = io.StringIO()
        buffer.write(report)

        st.download_button(
            label="Download Report",
            data=buffer.getvalue(),
            file_name=f"engine_{selected_engine}_report.txt",
            mime="text/plain"
        )