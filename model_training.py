# model_training.py
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mediation_model import TwoStageMediatorModel
from config import N_SPLITS, OUTPUT_DIR
import joblib

OUTPUT_DIR.mkdir(exist_ok=True)

def time_series_cv_train(df, features_social, features_controls, target_google='google_spend', target_revenue='revenue'):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold = 0
    results = []

    for train_idx, test_idx in tscv.split(df):
        fold += 1
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]

        model = TwoStageMediatorModel()
        model.fit(train, features_social, features_controls, target_google, target_revenue)

        y_pred = model.predict(test, features_social, features_controls)
        y_true = test[target_revenue].values

        # FIX: compute RMSE manually
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        results.append({'fold': fold, 'rmse': rmse, 'mae': mae})

        # Save fold predictions
        pred_df = test.copy()
        pred_df['predicted_revenue'] = y_pred
        pred_df.to_csv(OUTPUT_DIR / f'predictions_fold_{fold}.csv')

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'cv_results.csv', index=False)
    return results_df
