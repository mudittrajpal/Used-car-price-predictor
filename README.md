# Used Car Price Predictor 🚗

A solo end-to-end machine learning project that predicts used car prices on the German market using a dataset of ~370,000 listings from eBay Kleinanzeigen.

## Project Overview

| Step | File | Description |
|---|---|---|
| 1 | `step1_explore.py` | Load and explore raw data — shape, types, missing values, distributions |
| 2 | `step2_clean.py` | Remove outliers, handle missing values, encode categorical features |
| 3 | `step3_train.py` | Train Linear Regression and Random Forest models, evaluate with RMSE and R² |
| 4 | `step4_api.py` | Flask REST API — send car specs, receive predicted price |

## Dataset

[Germany Used Cars Dataset 2023](https://www.kaggle.com/datasets/wspirat/germany-used-cars-dataset-2023)  
~370,000 listings · German market · Downloaded from Kaggle

> The `data/` folder is gitignored (file size). Download `autos.csv` from the link above and place it in `data/autos.csv` to run this project.

## Setup

```bash
# Clone the repo
git clone https://github.com/mudittrajpal/Used-car-price-predictor.git
cd Used-car-price-predictor

# Install dependencies
pip install -r requirements.txt

# Run each step in order
python step1_explore.py
python step2_clean.py
python step3_train.py
python step4_api.py
```

## Tech Stack

- **Python** — pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Flask
- **Models** — Linear Regression (baseline), Random Forest Regressor
- **Evaluation** — RMSE, R²
- **Deployment** — REST API via Flask

## Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | €10,003 | €6,096 | 0.71 |
| Random Forest | €5,799 | €2,950 | 0.90 |