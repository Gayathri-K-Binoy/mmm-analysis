1. Data Preparation

Parse week into datetime and set as index.

Add seasonal features (weekofyear, month), moving averages (google_3w_ma, social_3w_ma).

Handle zero-spend periods explicitly with indicator flags.

Apply log transforms for skewed features (spends, revenue).

Standardize numeric features for regularized models.

2. Modeling Approach

Two-stage mediator-aware model:

Stage 1: Predict Google spend from social channels.

Stage 2: Predict revenue using predicted Google spend + social channels + controls.

Chosen models:

Ridge Regression (Stage 1)

ElasticNetCV (Stage 2) — interpretable with built-in feature selection.

Validation: Expanding-window time-series CV (no look-ahead).

3. Causal Framing

Explicitly model Google spend as a mediator (social → Google → revenue).

Avoid leakage by using predicted Google spend in Stage 2.

Acknowledge possible back-door paths (e.g., promotions affecting both social and revenue).

4. Diagnostics

Out-of-sample performance via RMSE and MAE per fold.

Residual plots, actual vs predicted plots.

Stability checks across folds.

5. Insights & Recommendations

Elasticities: interpret coefficients as marginal returns (when using ElasticNet).

Price elasticity: coefficients on average_price reflect sensitivity.

Promotion lift: quantified via promotion feature effect.

Risks:

Collinearity between social and Google (mediated effects may mask direct impacts).

Small sample size: coefficients may be unstable.

Practical advice: invest in social channels that most strongly stimulate Google search demand; monitor diminishing returns.