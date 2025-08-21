import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

def train_model(data: pd.DataFrame):
    """Train LightGBM model on given store's sales data."""
    # Features & target
    X = data[['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']]
    y = data['Weekly_Sales']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train model
    model = LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_test

def predict_future(data: pd.DataFrame, model, weeks: int = 12):
    """Predict next N weeks sales using last known values."""
    last_date = data['Date'].max()
    future_dates = pd.date_range(last_date, periods=weeks+1, freq='W')[1:]

    # Use last known feature values as baseline
    last_row = data.iloc[-1][['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']]
    future_features = pd.DataFrame([last_row.values] * weeks,
                                   columns=['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment'])

    preds = model.predict(future_features)

    return pd.DataFrame({'Date': future_dates, 'Prediction': preds})
