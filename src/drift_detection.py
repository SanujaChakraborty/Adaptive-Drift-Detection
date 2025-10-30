# src/drift_detection.py
import numpy as np
from scipy.stats import ks_2samp
from river.drift import ADWIN

def init_adwin(delta=0.01):
    return ADWIN(delta=delta)

def update_adwin(adwin, errors):
    drift_flag = False
    for e in errors:
        if adwin.update(e):   # returns True if drift detected
            drift_flag = True
            break
    return drift_flag


def compute_psi(expected, actual, bins=10):
    b = np.linspace(min(expected.min(), actual.min()), max(expected.max(), actual.max()), bins+1)
    e_perc = np.histogram(expected, bins=b)[0] / len(expected)
    a_perc = np.histogram(actual, bins=b)[0] / len(actual)
    e_perc = np.where(e_perc==0, 1e-6, e_perc)
    a_perc = np.where(a_perc==0, 1e-6, a_perc)
    psi = np.sum((e_perc - a_perc) * np.log(e_perc / a_perc))
    return psi

def ks_feature_changes(X_ref, X_window):
    ks_stats = {}
    for col in X_ref.columns:
        try:
            stat, p = ks_2samp(X_ref[col].values, X_window[col].values)
        except Exception:
            stat, p = 0.0, 1.0
        ks_stats[col] = (stat, p)
    return ks_stats
