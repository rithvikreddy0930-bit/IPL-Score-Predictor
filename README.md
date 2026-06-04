# IPL Score Predictor

A machine learning project that predicts first-innings IPL scores using historical ball-by-ball match data.

## Project Overview

This project uses IPL match data to estimate the final first-innings score based on the current match situation. The goal is to explore data preprocessing, feature engineering, and regression-based machine learning models for score prediction.

## Dataset

- Historical IPL ball-by-ball match data
- Match-level and innings-level information
- Features engineered from live match state

## Features Used

Some of the key features include:

- Current score
- Wickets fallen
- Balls completed
- Current run rate
- Recent scoring trends
- Partnership information
- Team and venue related features

## Machine Learning Workflow

1. Data Cleaning and Preprocessing
2. Feature Engineering
3. Exploratory Data Analysis (EDA)
4. Model Training
5. Model Evaluation
6. Performance Comparison

## Models Experimented With

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

## Results

Current best model performance:

- R² Score: ~0.70

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn

## Future Improvements

- Additional feature engineering
- Hyperparameter tuning
- Better handling of match context
- Testing on larger datasets
- Model deployment as a web application

## Author

Rithvik Reddy
