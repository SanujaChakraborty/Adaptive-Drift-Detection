import pandas as pd
import numpy as np
import time, psutil, os
from scipy.stats import ttest_rel

# ----------------- Helper: full retraining -----------------
def measure_retrain_full(clf_factory, X_ref, y_ref, X_win, y_win):
    proc = psutil.Process()
    t0 = time.time()
    mem0 = proc.memory_info().rss
    clf = clf_factory()
    X_full = pd.concat([X_ref, X_win])
    y_full = pd.concat([y_ref, y_win])
    clf.fit(X_full, y_full)
    t1 = time.time()
    mem1 = proc.memory_info().rss
    return round(t1 - t0, 4), round((mem1 - mem0) / 1024 / 1024, 4)

# ----------------- Helper: selective retraining -----------------
def measure_retrain_selective(clf_factory, X_ref, y_ref, X_win, y_win, selected_features):
    proc = psutil.Process()
    t0 = time.time()
    mem0 = proc.memory_info().rss
    clf = clf_factory()
    X_retrain = pd.concat([X_ref[selected_features], X_win[selected_features]])
    y_retrain = pd.concat([y_ref, y_win])
    clf.fit(X_retrain, y_retrain)
    t1 = time.time()
    mem1 = proc.memory_info().rss
    return round(t1 - t0, 4), round((mem1 - mem0) / 1024 / 1024, 4)

# ----------------- Main comparison function -----------------
def compare_full_vs_selective(clf_factory, X_ref, y_ref, X_win, y_win, top_k, compute_psi, repeats=5):
    times_full, mems_full = [], []
    times_sel, mems_sel = [], []

    for _ in range(repeats):
        # Full retrain
        t_full, m_full = measure_retrain_full(clf_factory, X_ref, y_ref, X_win, y_win)
        times_full.append(t_full); mems_full.append(m_full)

        # Selective retrain based on top-k PSI features
        psi_vals = {c: compute_psi(X_ref[c].values, X_win[c].values) for c in X_ref.columns}
        selected = sorted(psi_vals, key=psi_vals.get, reverse=True)[:top_k]
        t_sel, m_sel = measure_retrain_selective(clf_factory, X_ref, y_ref, X_win, y_win, selected)
        times_sel.append(t_sel); mems_sel.append(m_sel)

    # Summary stats
    def summary(arr): return f"{np.mean(arr):.3f} ± {np.std(arr):.3f}"
    print("Full retrain time:", summary(times_full))
    print("Selective retrain time:", summary(times_sel))
    print("Full memory:", summary(mems_full))
    print("Selective memory:", summary(mems_sel))

    # Statistical test
    t_t, p_t = ttest_rel(times_full, times_sel)
    mem_t, mem_p = ttest_rel(mems_full, mems_sel)
    print(f"Paired t-test: time p={p_t:.4f}, memory p={mem_p:.4f}")

    # ----------------- Save results -----------------
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame({
        "retrain_type": ["Full"] * repeats + ["Selective"] * repeats,
        "time_s": times_full + times_sel,
        "memory_mb": mems_full + mems_sel
    })

    df["run"] = list(range(1, repeats + 1)) * 2
    df.to_csv("results/compare_retraining.csv", index=False)

    print("📂 Saved retraining efficiency comparison to results/compare_retraining.csv")

    return df
