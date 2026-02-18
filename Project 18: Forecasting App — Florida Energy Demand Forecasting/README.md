# Project 18: Forecasting App — Florida Energy Demand Forecasting

## Situation / Objective
Energy demand forecasting supports planning, capacity decisions, and operational readiness. The objective of this project is to build an end-to-end, reproducible forecasting workflow for Florida energy demand with:

- A modular training/prediction pipeline (`src/`)
- A Streamlit UI for interactive usage
- Docker packaging for portable deployment

## Task
- Set up a local Python environment.
- Run training + forecasting from the codebase.
- Run a Streamlit UI for interactive forecasts.
- Package and run the app via Docker.

## Actions
- Organized the project as a small Python application:
  - Training and prediction code in `src/`
  - UI entry point in `app.py`
  - Reproducible dependencies in `requirements.txt`
  - Containerization via `Dockerfile`

- Implemented common dev workflows:
  - Local environment creation and dependency install
  - CLI-style run for training
  - Streamlit run for UI
  - Docker build/run for deployment-like execution

## Results / Summary
- Produced a runnable forecasting project that can be executed locally or packaged into a container.
- Established a clean, modular structure that supports iteration (EDA → features → training → prediction → UI).

## How to run

### Local setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Run training + forecasting locally

```bash
python -m src.train
```

### Run Streamlit UI

```bash
streamlit run app.py
```

### Docker build & run
```bash
# to remove the image if already built
docker ps -a
docker stop energy_app
docker rm energy_app
```

# to create the image
docker build -t fl-energy-forecast .  
# to run the image
docker run -p 8501:8501 --name energy_app fl-energy-forecast
```
