# Predictive Maintenance Agent

## 📌 Project Overview

This project focuses on building an intelligent Predictive Maintenance (PdM) system for aircraft engines using degradation-based sensor data.

The objective is to analyze engine telemetry data, understand degradation patterns, and progressively develop a Remaining Useful Life (RUL) prediction system.

This repository documents the engineering evolution of the system phase by phase.

---

## 🚀 Current Phase: Phase 1 — Data Exploration & Understanding

In this phase, the focus is on:

- Understanding the CMAPSS turbofan engine dataset
- Exploring degradation behavior across engine cycles
- Visualizing sensor trends
- Identifying relevant features for RUL prediction

All exploration work is available in:

```
notebooks/notebooks_01_data_exploration.ipynb
```

---

## 📊 Dataset Information

This project uses the NASA CMAPSS Turbofan Engine Degradation Simulation Dataset (FD001 subset).

Dataset characteristics:

- Multiple engines operating under different cycles
- 21 sensor measurements per cycle
- Gradual degradation over time
- Goal: Predict Remaining Useful Life (RUL)

Dataset files are stored under:

```
data/raw/
```

---

## 📁 Project Structure

```
predictive-maintenance-agent/
│
├── data/
│   └── raw/                # Raw dataset files
│
├── notebooks/
│   └── notebooks_01_data_exploration.ipynb
│
├── src/
│   └── __init__.py         # Source modules (to be expanded in later phases)
│
├── docs/                   # Documentation
│
└── README.md
```

---

## 🗺 Project Roadmap

The system will evolve through structured phases:

- Phase 1: Data Exploration & Understanding ✅
- Phase 2: Feature Engineering
- Phase 3: Model Training & RUL Prediction
- Phase 4: Evaluation & Visualization
- Phase 5: System Integration & Deployment

Each phase will be committed progressively to demonstrate system evolution.

---

## 🧠 Long-Term Goal

To build a scalable predictive maintenance pipeline capable of:

- Processing engine telemetry data
- Learning degradation patterns
- Predicting RUL
- Serving predictions via backend APIs
- Enabling monitoring dashboards

---

## 👩‍💻 Author

Shifanaaz Abdulsab Nadaf  
B.E. Computer Science & Engineering  
KLE Technological University