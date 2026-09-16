---
title: Hotel Revenue Optimizer
emoji: 🏨
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.40.0
app_file: app/revenue_app.py
pinned: false
---

# Hotel Revenue Optimizer

A machine learning app that predicts hotel booking cancellations and revenue outcomes, helping hotels make data-driven decisions around pricing and inventory.

**Live demo:** [Hugging Face Spaces](https://huggingface.co/spaces/akshayakeerthi20/hotel-revenue-optimizer)

## Overview

The app uses a Random Forest model trained on historical hotel booking data to:
- Predict the likelihood of a booking cancellation
- Estimate expected revenue outcomes based on booking characteristics

## Model Performance

| Metric | Value |
|--------|-------|
| MAE    | 16.56 |
| R²     | 0.76  |

## Project Structure

```
hotel-booking-ml/
├── app/                    # Streamlit application
│   └── revenue_app.py
├── data/
│   └── processed/          # Cleaned, model-ready dataset (Git LFS)
├── models/                 # Trained model artifacts (Git LFS)
│   ├── revenue_model.pkl
│   ├── revenue_features.pkl
│   └── revenue_scaler.pkl
├── notebooks/               # Pipeline notebooks, run in order
│   ├── 01_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_revenue_model.ipynb
├── outputs/
│   └── figures/             # Saved plots and visualizations
├── requirements.txt
└── README.md
```

## Setup

Clone the repo and install dependencies:

```bash
git clone https://github.com/Akshayakeerthi20/hotel-revenue-optimizer.git
cd hotel-revenue-optimizer
pip install -r requirements.txt
```

Run the app locally:

```bash
streamlit run app/revenue_app.py
```

## Model Details

The final model is a `RandomForestRegressor` tuned with a capped tree depth and minimum leaf size to keep the model compact and generalizable:

```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)
```

Model artifacts are versioned with Git LFS due to their size.

## Notebooks

Notebooks are numbered to reflect the pipeline order — run them sequentially if reproducing results from raw data:

1. `01_cleaning.ipynb` — data cleaning and preprocessing
2. `02_eda.ipynb` — exploratory data analysis
3. `03_modeling.ipynb` — initial model experimentation
4. `04_revenue_model.ipynb` — final revenue model training and evaluation