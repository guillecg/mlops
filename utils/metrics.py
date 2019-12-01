import numpy as np
import pandas as pd

from sklearn.metrics import *


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def get_metrics(sets_dict):
    metrics_df = pd.DataFrame()

    for set_key, tupla in sets_dict.items():
        custom_metrics = {
            'R2': r2_score(*tupla),
            'Explained variance': explained_variance_score(*tupla),
            'Max error': max_error(*tupla),
            'Mean absolute error': mean_absolute_error(*tupla),
            'Median absolute error': median_absolute_error(*tupla),
            'Mean absolute precentage error': \
                mean_absolute_percentage_error(*tupla),
            'Mean squared error': mean_squared_error(*tupla),
            'Root mean squared error': np.sqrt(mean_squared_error(*tupla)),
        }

        row_df = pd.Series(custom_metrics).to_frame().T.round(4)
        row_df.rename(columns={0: 'Score'}, inplace=True)

        metrics_df = metrics_df.append(row_df)

    metrics_df.index = list(sets_dict.keys())

    return metrics_df.T

