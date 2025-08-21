import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from model import train_model, predict_future
from utils import preprocess_data
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Streamlit page setup
st.set_page_config(page_title="Walmart Sales Prediction", layout="wide")
st.title("🛒 Walmart Sales Prediction Dashboard")

# Load and preprocess dataset
data = pd.read_csv("data/walmart_sales.csv", parse_dates=['Date'])
data = preprocess_data(data)

# Sidebar filters
st.sidebar.header("Filter Options")
store = st.sidebar.selectbox("Select Store", data['Store'].unique())
filtered_data = data[data['Store'] == store]

# Display historical sales
st.subheader("Historical Sales")
fig_hist = px.line(filtered_data, x='Date', y='Weekly_Sales', title=f'Weekly Sales - Store {store}')
st.plotly_chart(fig_hist, use_container_width=True)

# Train model
st.subheader("Model Training")
model, X_train, X_test, y_test = train_model(filtered_data)
st.success("✅ Model trained successfully!")

# Evaluate
y_pred = model.predict(X_test)
st.write("📊 Model Evaluation:")
st.write("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
mse = mean_squared_error(y_test, y_pred)
rmse = float(np.sqrt(mse))
st.write("RMSE:", round(rmse, 2))


# Predict future sales
st.subheader("🔮 Predict Next 12 Weeks Sales")
future_df = predict_future(filtered_data, model)
fig_pred = px.line(future_df, x='Date', y='Prediction', title=f'Predicted Weekly Sales - Store {store}')
st.plotly_chart(fig_pred, use_container_width=True)
