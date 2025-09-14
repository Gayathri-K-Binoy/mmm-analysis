# main.py
from data_preprocessing import load_and_preprocess
from model_training import time_series_cv_train
from evaluate import residual_plot, plot_true_vs_pred, save_metrics
import pandas as pd
from config import OUTPUT_DIR

OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    # Step 1: Load & preprocess data
    df = load_and_preprocess()

    # Step 2: Define features
    features_social = ['facebook_spend', 'tiktok_spend', 'instagram_spend', 'snapchat_spend']
    features_controls = ['average_price', 'promotions', 'emails_send', 'sms_send', 'social_followers']

    # Step 3: Train with time-series CV
    results = time_series_cv_train(df, features_social, features_controls)
    save_metrics(results)
    print('CV results:')
    print(results)

    # Step 4: Fit final model on full data
    from mediation_model import TwoStageMediatorModel
    model = TwoStageMediatorModel()
    model.fit(df, features_social, features_controls)
    preds = model.predict(df, features_social, features_controls)

    # Step 5: Diagnostics plots
    residual_plot(df['revenue'].values, preds, filename='residuals_full.png')
    plot_true_vs_pred(df['revenue'].values, preds, filename='true_vs_pred_full.png')

    # Step 6: Save final predictions
    df_out = df.copy()
    df_out['predicted_revenue'] = preds
    df_out.to_csv(OUTPUT_DIR / 'predictions_full.csv')
    print('Outputs saved to outputs/')

if __name__ == '__main__':
    main()
