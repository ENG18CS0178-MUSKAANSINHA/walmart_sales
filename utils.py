import pandas as pd

def preprocess_data(df):
    # Parse date with dayfirst=True since your format is DD-MM-YYYY
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    # Extract useful time features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week

    return df
