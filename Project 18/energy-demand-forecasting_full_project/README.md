# Florida Energy Demand Forecasting

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Run training + forecasting locally

```bash
python -m src.train
```

## Run Streamlit UI

```bash
streamlit run app.py
```

## Docker build & run
```bash 
# to remove the image if already build
docker ps -a                        
docker stop energy_app                                      
docker rm energy_app
```

```bash
# to create the image
docker build -t fl-energy-forecast .                                                              
docker run -p 8501:8501 --name energy_app fl-energy-forecast
```
