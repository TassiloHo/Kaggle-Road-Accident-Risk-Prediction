# 🚗 Road Accident Risk Prediction

This repository contains a machine learning solution for predicting road accident risk using XGBoost, developed for the Kaggle Playground Series S5E10 competition.

## 🔗 Links

- **Kaggle Profile**: [tussalo](https://www.kaggle.com/tussalo)
- **Live Streamlit App**: [Road Accident Prediction Game](https://road-accident-prediction-kaggle-s5e10.streamlit.app/)
- **Kaggle Competition**: [Playground Series S5E10](https://www.kaggle.com/competitions/playground-series-s5e10)

## 🛠 Features

### Model Training and Submission(`fit_xgboost.ipynb`)
- assumes kaggle credentials have been set up for api use
- Data preprocessing with categorical encoding and memory optimization
- XGBoost regressor with categorical feature support
- Hyperparameter optimization using Optuna
- Cross-validation with RepeatedKFold (10 splits, 2 repeats)
- Automatic Kaggle submission

### Interactive Web App (`app.py`)
- **Prediction Game**: Challenge yourself against the AI model
- **Real-time Scoring**: Track your performance vs the model
- **SHAP Explanations**: Understand model predictions with waterfall plots
- **Visual Comparisons**: See prediction accuracy side-by-side
