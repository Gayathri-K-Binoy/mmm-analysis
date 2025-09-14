# config.py
import pathlib


SEED = 42
DATA_PATH = pathlib.Path('marketing_data.csv')
OUTPUT_DIR = pathlib.Path('outputs')
N_SPLITS = 5 # number of folds for time-series CV (expanding-window)
RANDOM_STATE = SEED
SMALL_CONST = 1e-6 # for log transforms