# evaluate.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config import OUTPUT_DIR

OUTPUT_DIR.mkdir(exist_ok=True)

def residual_plot(y_true, y_pred, filename='residuals.png'):
    residuals = y_true - y_pred
    plt.figure(figsize=(8,4))
    plt.scatter(y_pred, residuals)
    plt.axhline(0, linestyle='--')
    plt.xlabel('Predicted Revenue')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Revenue')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()

def plot_true_vs_pred(y_true, y_pred, filename='true_vs_pred.png'):
    plt.figure(figsize=(8,4))
    plt.plot(y_true, label='True Revenue')
    plt.plot(y_pred, label='Predicted Revenue')
    plt.legend()
    plt.title('True vs Predicted Revenue')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()

def save_metrics(df_results, filename='metrics_summary.txt'):
    """Save CV results dataframe to a text file"""
    with open(OUTPUT_DIR / filename, 'w') as f:
        f.write(df_results.to_string(index=False))
