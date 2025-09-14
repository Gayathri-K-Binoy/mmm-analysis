# data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import DATA_PATH, SMALL_CONST

def load_and_preprocess(path=DATA_PATH, dropna=True):
    df = pd.read_csv(path)

    # Parse week column
    if 'week' in df.columns:
        # Attempt common formats
       df['week'] = pd.to_datetime(df['week'], format="%Y-%m-%d", errors='coerce')

    else:
        raise ValueError('CSV must contain a "week" column')

    df = df.sort_values('week').reset_index(drop=True)
    df = df.set_index('week')

    # Fill or drop rows with invalid dates
    if dropna:
        df = df[~df.index.isna()].copy()

    # Feature engineering: week of year for seasonality
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['month'] = df.index.month

    # Ensure spend columns exist
    spend_cols = ['facebook_spend','google_spend','tiktok_spend','instagram_spend','snapchat_spend']
    for c in spend_cols:
        if c not in df.columns:
            df[c] = 0.0

    # Zero-spend indicators
    for c in spend_cols:
        df[f'{c}_is_zero'] = (df[c] <= 0).astype(int)

    # Moving averages to capture trend
    df['google_3w_ma'] = df['google_spend'].rolling(3, min_periods=1).mean()
    df['total_social_spend'] = df[['facebook_spend','tiktok_spend','instagram_spend','snapchat_spend']].sum(axis=1)
    df['social_3w_ma'] = df[['facebook_spend','tiktok_spend','instagram_spend','snapchat_spend']].sum(axis=1).rolling(3, min_periods=1).mean()

    # Log transform skewed positive variables (spends, emails, revenue)
    log_cols = ['facebook_spend','google_spend','tiktok_spend','instagram_spend','snapchat_spend','emails_send','sms_send','revenue','total_social_spend']
    for c in log_cols:
        if c in df.columns:
            df[f'log_{c}'] = np.log(df[c].clip(lower=SMALL_CONST) + SMALL_CONST)

    # Standard scaling of numeric features for linear models
    numeric_cols = ['average_price','social_followers','promotions','weekofyear','month','google_3w_ma','social_3w_ma']
    scaler = StandardScaler()
    present_numeric = [c for c in numeric_cols if c in df.columns]
    if present_numeric:
        df[[f'std_{c}' for c in present_numeric]] = scaler.fit_transform(df[present_numeric])

    return df


if __name__ == '__main__':
    df = load_and_preprocess()
    print(df.head())