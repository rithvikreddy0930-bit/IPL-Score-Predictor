# IPL First Innings Score Predictor

A machine learning project that predicts the **final first-innings score in IPL matches** using historical ball-by-ball data.

The model estimates what score a batting team is likely to finish with based on the current match state, momentum, player information, and match context.

---

## Overview

This project focuses on building a realistic IPL score prediction system using:

- Data cleaning and preprocessing
- Cricket-specific feature engineering
- Leakage-free model evaluation
- Hyperparameter tuning
- Feature importance analysis
- Error analysis

Only first-innings data is used.

---

## Dataset

- Historical IPL ball-by-ball data
- Seasons: **2008–2026**
- One row per delivery
- Target variable: `final_score`

Dataset location:

```text
data/IPL.csv
```

---

## Features Used

### Match State

- `team_runs`
- `team_balls`
- `team_wicket`
- `current_run_rate`
- `balls_remaining`
- `wickets_remaining`

### Momentum Features

- `last_5_over_runs`
- `last_2_over_runs`
- `last_5_over_wickets`
- `last_3_over_wickets`

### Partnership & Player Features

- `current_partnership`
- `striker_career_sr`
- `non_striker_sr`

### Match Context

- `batting_team`
- `bowling_team`
- `venue`
- `season`
- `stage`

Categorical features are one-hot encoded.

---

## Preventing Data Leakage

Since multiple rows belong to the same match, a standard random train-test split would leak information between training and testing data.

To avoid this, the project uses:

```python
GroupShuffleSplit(groups=match_id)
```

This ensures that deliveries from the same match never appear in both train and test sets.

---

## Model Development

The following models were evaluated:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

Hyperparameter tuning was performed using `RandomizedSearchCV`.

The tuned XGBoost model achieved the best performance and was selected as the final model.

---

## Deep Learning Experiment

A feed-forward Artificial Neural Network (ANN) was also evaluated using TensorFlow/Keras.

The ANN pipeline included:

- Feature scaling
- Dense hidden layers
- Dropout regularization
- EarlyStopping to reduce overfitting

### ANN Results

| Metric | Value |
|----------|----------|
| R² Score | **0.624** |
| MAE | **15.15** |
| MSE | **380.46** |

The ANN improved over baseline models but was outperformed by the tuned XGBoost model.

This suggests that gradient-boosted trees were better suited for this feature-engineered tabular cricket dataset.

---

## Final Results

### XGBoost Regressor

| Metric | Value |
|----------|----------|
| R² Score | **0.705** |
| MAE | **13.12** |
| MSE | **298.60** |

### Most Important Features

- Current Run Rate
- Last 5 Over Runs
- Team Runs
- Wickets Lost
- Balls Remaining
- Current Partnership

These features capture both the current match state and recent scoring momentum.

---

## Key Findings

- Recent scoring momentum is one of the strongest predictors of final score.
- Career strike-rate features provide small but useful predictive signal.
- Team recent-form features did not improve performance and were removed.
- The model performs best on modern IPL seasons, where scoring patterns are more consistent.

---

## Error Analysis

The model performs well on typical IPL scores but struggles with rare extreme outcomes.

- Extremely low scores tend to be overpredicted.
- Extremely high scores tend to be underpredicted.

This regression-to-the-mean behaviour occurs because extreme matches are relatively rare in the training data.

---

## Project Structure

```text
IPL-Score-Predictor/
│
├── data/
│   └── IPL.csv
│
├── models/
│   └── ipl_score_predictor.pkl
│
├── src/
│   └── ipl_score_predictor.py
│
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
python src/ipl_score_predictor.py
```

Place the trained model inside:

```text
models/ipl_score_predictor.pkl
```

before running predictions.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib

---

## Future Improvements

- Batter-bowler matchup statistics
- Advanced player form metrics
- Deep learning sequence models
- Live score prediction interface
- Web deployment

---

## Author

**Rithvik Reddy**