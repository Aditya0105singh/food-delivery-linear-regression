# Food Delivery Time Prediction

This project implements multiple linear regression models to predict food delivery times based on various factors like distance, weather, traffic conditions, and courier experience.

## Features

- **Data Loading**: Robust CSV data loading with error handling
- **Data Cleaning**: Missing value imputation and categorical encoding
- **Multiple Models**: Linear, Ridge, and Lasso regression
- **Evaluation**: Comprehensive metrics (RMSE, MAE, R²)
- **Modular Design**: Clean separation of concerns

## Dataset

- **Food_Delivery_Times.csv**: 1000 delivery records
- **Target**: Delivery_Time_min (delivery time in minutes)
- **Features**: Distance, Weather, Traffic, Time of Day, Vehicle Type, etc.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
cd src
python main.py
```

## Project Structure

```
food-delivery-linear-regression/
│
├── data/
│   └── Food_Delivery_Times.csv
│
├── src/
│   ├── data_loader.py      # Data loading utilities
│   ├── data_cleaner.py     # Data preprocessing
│   ├── regression_models.py # ML models
│   └── main.py            # Training pipeline
│
├── requirements.txt
└── README.md
```

## Results

The pipeline evaluates three regression models and selects the best performing one based on R² score.

## Future Improvements

- Add hyperparameter tuning
- Include more advanced models
- Add visualization capabilities
- Deploy as web service
