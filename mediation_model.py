# mediation_model.py
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from config import OUTPUT_DIR, RANDOM_STATE
import os

OUTPUT_DIR.mkdir(exist_ok=True)

class TwoStageMediatorModel:
    """
    Stage 1: model google_spend ~ socials + controls
    Stage 2: model revenue ~ predicted_google + socials + controls

    This class uses Ridge for Stage 1 and ElasticNetCV for Stage 2 by default.
    """

    def __init__(self, stage1_model=None, stage2_model=None):
        self.stage1_model = stage1_model or Ridge(random_state=RANDOM_STATE)
        self.stage2_model = stage2_model or ElasticNetCV(cv=5, random_state=RANDOM_STATE)

    def fit(self, df, features_social, features_controls, target_google='google_spend', target_revenue='revenue'):
        # Stage 1
        X1 = df[features_social + features_controls].fillna(0)
        y1 = df[target_google].fillna(0)
        self.stage1_model.fit(X1, y1)
        df = df.copy()
        df['google_hat'] = self.stage1_model.predict(X1)

        # Stage 2 - use google_hat and the same controls and social features
        X2 = df[['google_hat'] + features_social + features_controls].fillna(0)
        y2 = df[target_revenue].fillna(0)
        self.stage2_model.fit(X2, y2)

        # Save
        joblib.dump(self.stage1_model, OUTPUT_DIR / 'stage1_model.joblib')
        joblib.dump(self.stage2_model, OUTPUT_DIR / 'stage2_model.joblib')

    def predict(self, df, features_social, features_controls):
        X1 = df[features_social + features_controls].fillna(0)
        google_hat = self.stage1_model.predict(X1)
        X2 = pd.DataFrame({'google_hat': google_hat})
        for c in features_social + features_controls:
            X2[c] = df[c].values
        pred = self.stage2_model.predict(X2.fillna(0))
        return pred

    def get_stage2_coefs(self, feature_names):
        # Only valid for linear stage2_model with coef_ attribute
        if hasattr(self.stage2_model, 'coef_'):
            return dict(zip(feature_names, self.stage2_model.coef_))
        return None