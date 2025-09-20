# ✈️ Flight Delay Prediction

This project builds machine learning models to predict flight delays based on flight schedule, aircraft, trajectory, and airport information.

Delays is a crucial problem for tunisair, an airline that usually receives backlash due to its poor services. By predicting delays in advance we can improve customer satisfaction.

# Dataset

the dataset was obtained from kaggle:
[dataset link](https://www.kaggle.com/datasets/abderrahimalakouche/flight-delay-prediction)

The dataset contains flight-level records with the following important features:

+ Flight information: FLTID, AC (aircraft), trajectory
+ Airports: DEPSTN (departure), ARRSTN (arrival)
+ Schedule: STD (scheduled departure), STA (scheduled arrival)
+ Date features: DATOP (operating date), season

The target variable is delay in minutes.
To stabilize variance, delays are log-transformed using ``np.log1p(delay_minutes)`` during training.

# Preprocessing

+ Handle missing values

+ Generate time-based features: dep_hour, arr_hour, day_of_week, day_of_year, flight_duration_min
+ + Encode categorical features:
+ + For XGBoost: TargetEncoder + OrdinalEncoder
+ + For CatBoost: native categorical support (cast to str)
+ sample weighting to emphasize middle ranges (40–140 minutes)

## Models

I experimented with multiple approaches:

+ Baseline Models

+ + Linear Regression / SGDRegressor — fast, but poor fit due to non-linear relationships.

+ Gradient Boosting

+ + XGBoost (XGBRegressor)

+ + + Strong performance with feature engineering

+ + + Sensitive to hyperparameters → tuned with RandomizedSearchCV

+ + + Handles imbalanced target via sample weights

+ + CatBoost (CatBoostRegressor)

+ + + Handles high-cardinality categoricals (trajectory, FLTID) natively

+ + + Robust to skewed data

+ + + Supports quantile regression for uncertainty intervals

## Evaluation

+ Main metrics:

+ + RMSE (minutes)

+ + MAE (minutes)

Both computed after inverting log transformation (``np.expm1``).

Performance is also broken down by delay ranges:

+ Low: 0–40 minutes
+ Medium: 40–140 minutes
+ High: >140 minutes

## Results

XGBoost achieved RMSE ≈ 45.895  minutes

CatBoost achieved RMSE ≈ 1.375 (better on medium delays)

Adding sample weights for 40–140 min delays improved accuracy in the middle range while keeping global RMSE stable.
