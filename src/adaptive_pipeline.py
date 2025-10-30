# src/adaptive_pipeline.py
import os, time, psutil
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from load_data import load_nsl_kdd, load_cicids
from feature_analysis import preprocess
from stream_generator import stream_windows
from drift_detection import init_adwin, update_adwin, compute_psi
from explainability import explain_model
from measure_perf import compare_full_vs_selective

# ---------------- Adaptive Top-K Calculation ----------------
def calculate_adaptive_top_k(total_features, psi_scores=None, method="percentage"):
    """
    Calculate optimal top-k features adaptively
    Methods: 
    - "percentage": top 25% of features
    - "threshold": features with PSI > 0.2
    - "statistical": features with PSI > mean + 1*std
    """
    if method == "percentage":
        # Use top 25% of features, with bounds [5, 20]
        k = int(0.25 * total_features)
        k = max(5, min(k, 20))  # Keep between 5 and 20 features
        
    elif method == "threshold" and psi_scores is not None:
        # Use features with PSI > 0.2 (significant drift)
        k = len([psi for psi in psi_scores.values() if psi > 0.2])
        k = max(5, min(k, total_features))  # Ensure reasonable bounds
        
    elif method == "statistical" and psi_scores is not None:
        # Use statistically significant drifted features
        psi_values = list(psi_scores.values())
        if len(psi_values) > 0:
            mean_psi = sum(psi_values) / len(psi_values)
            std_psi = (sum((x - mean_psi) ** 2 for x in psi_values) / len(psi_values)) ** 0.5
            threshold = mean_psi + std_psi
            k = len([psi for psi in psi_scores.values() if psi > threshold])
            k = max(5, min(k, total_features))
        else:
            k = min(10, total_features)  # Fallback
    else:
        # Default fallback
        k = min(10, total_features)
    
    print(f"🔧 Adaptive top-k selected: {k} features (method: {method})")
    return k

# ---------------- Model factory ----------------
def get_model(model_name="RandomForest"):
    if model_name == "RandomForest":
        return RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
    elif model_name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, solver='lbfgs')
    elif model_name == "SVM":
        return SVC(probability=True)
    elif model_name == "XGBoost":
        return XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError(f"Unknown model: {model_name}")

# ---------------- MAIN PIPELINE ----------------
def run_pipeline(dataset="nsl", path=None, model_name="RandomForest",
                 init_size=3000, window_size=1000, top_k_method="percentage"):
    
    # ---------------- Load Dataset ----------------
    if dataset == "nsl":
        print(f"Loading NSL-KDD from {path} ...")
        df = load_nsl_kdd(path, nrows=20000)
    elif dataset == "cicids":
        print(f"Loading CICIDS from {path} ...")
        df = load_cicids(path, nrows=20000)
    else:
        raise ValueError("Dataset must be 'nsl' or 'cicids'")

    print("Data shape:", df.shape)

    # ---------------- Preprocess ----------------
    X, y = preprocess(df)

    # ➕ Skip single-class files (avoid ValueError)
    if len(y.unique()) < 2:
        print(f"⚠️ Skipping {path} — only one class present: {y.unique()}")
        return

    # ------------------------------------------------------------
    # Step 1️⃣  Ensure initial training window has ≥2 classes
    # ------------------------------------------------------------
    y_unique = y.unique()
    if len(y_unique) < 2:
        raise ValueError(f"Dataset has only one class ({y_unique})! Cannot train model.")

    # Try a stratified split, fallback if single class issue appears
    try:
        X_init, _, y_init, _ = train_test_split(
            X, y, train_size=init_size, stratify=y, random_state=42
        )
    except ValueError:
        print("⚠️ Stratified split failed (single class in this file). Creating balanced sample manually...")
        # Identify all classes
        class_counts = y.value_counts()
        print("Class distribution:", class_counts.to_dict())

        # If there's only one class, we need to mix in a second from deeper in the dataset
        if len(class_counts) == 1:
            first_class = class_counts.index[0]
            # Pick first 3000 samples of that class
            X_class = X[y == first_class].iloc[:1500]
            y_class = y.loc[X_class.index]

            # Find another class (if exists) from the dataset
            alt_classes = y[y != first_class].index
            if len(alt_classes) > 0:
                X_other = X.loc[alt_classes].sample(n=min(1500, len(alt_classes)), random_state=42)
                y_other = y.loc[X_other.index]
                X_init = pd.concat([X_class, X_other])
                y_init = pd.concat([y_class, y_other])
            else:
                raise ValueError("❌ No secondary class found in the file. Try another CICIDS day file.")
        else:
            # Both classes exist, sample balanced
            minority_class = class_counts.idxmin()
            majority_class = class_counts.idxmax()
            n = min(1500, len(X[y == minority_class]), len(X[y == majority_class]))
            X_min = X[y == minority_class].sample(n, random_state=42)
            X_maj = X[y == majority_class].sample(n, random_state=42)
            X_init = pd.concat([X_min, X_maj])
            y_init = pd.concat([y.loc[X_min.index], y.loc[X_maj.index]])

    print(f"✅ Initial training window prepared with class counts: {y_init.value_counts().to_dict()}")

    # Calculate adaptive top-k based on total features
    total_features = X.shape[1]
    adaptive_top_k = calculate_adaptive_top_k(total_features, method=top_k_method)
    
    clf = get_model(model_name)
    clf.fit(X_init, y_init)
    print(f"✅ Baseline trained on {len(X_init)} samples using {model_name}.")

    current_features = X.columns.tolist()
    adwin = init_adwin(delta=0.002)
    proc = psutil.Process(os.getpid())
    ref_X, ref_y = X_init.copy(), y_init.copy()
    idx, results = 0, []

    # ---------------- Streaming ----------------
    for X_win, y_win in stream_windows(X, y, init_size=init_size, window_size=window_size):
        preds = clf.predict(X_win[current_features])
        errors = (preds != y_win).astype(int)
        drift_flag = update_adwin(adwin, errors)

        # Force drift for first window for demo
        if idx == 0:
            drift_flag = True

        acc = accuracy_score(y_win, preds)
        f1 = f1_score(y_win, preds, average='weighted', zero_division=0)
        print(f"\n🔹 Window {idx}: acc={acc:.4f}, f1={f1:.4f}, drift={drift_flag}")

        if drift_flag:
            print("\n🚨 Drift detected! Retraining...")

            # Calculate PSI values for all features
            psi_vals = {c: compute_psi(ref_X[c].values, X_win[c].values) for c in ref_X.columns}
            
            # Use adaptive top-k based on PSI scores
            current_top_k = calculate_adaptive_top_k(total_features, psi_vals, method=top_k_method)
            
            # Select top-k features by PSI
            selected_features = sorted(psi_vals, key=psi_vals.get, reverse=True)[:current_top_k]
            current_features = selected_features

            # Optional performance comparison: full vs selective retraining
            try:
                compare_full_vs_selective(
                    clf_factory=lambda: get_model(model_name),
                    X_ref=ref_X,
                    y_ref=ref_y,
                    X_win=X_win,
                    y_win=y_win,
                    top_k=current_top_k,
                    compute_psi=compute_psi
                )
            except Exception as e:
                print(f"⚠️ Skipped compare_full_vs_selective: {e}")

            t0 = time.time()
            mem_before = proc.memory_info().rss / 1024 / 1024

            X_retrain = pd.concat([ref_X[selected_features], X_win[selected_features]], axis=0)
            y_retrain = pd.concat([ref_y, y_win], axis=0)

            clf = get_model(model_name)
            clf.fit(X_retrain, y_retrain)

            t1 = time.time()
            mem_after = proc.memory_info().rss / 1024 / 1024
            retrain_time = round(t1 - t0, 2)
            mem_diff = round(mem_after - mem_before, 2)
            print(f"✅ Retraining done in {retrain_time}s | Memory change: {mem_diff} MB")
            print(f"🎯 Used {len(selected_features)} features: {selected_features[:5]}...")  # Show first 5 features

            # SHAP explainability per window
            try:
                import matplotlib.pyplot as plt
                sample_size = min(100, len(X_win), len(current_features))
                X_sample = X_win[current_features].sample(sample_size, random_state=42)

                fig = explain_model(clf, X_sample, model_type=model_name)
                if fig:
                    os.makedirs("results/shap_per_window", exist_ok=True)
                    fig.savefig(f"results/shap_per_window/{dataset}_{model_name}_window{idx}_shap.png", bbox_inches='tight')
                plt.close('all')
            except Exception as e:
                print(f"⚠️ SHAP skipped for window {idx}: {e}")

            # Evaluate performance after retraining
            preds2 = clf.predict(X_win[current_features])
            acc2 = accuracy_score(y_win, preds2)
            f12 = f1_score(y_win, preds2, average='weighted', zero_division=0)

            # Log results
            results.append({
                'window': idx,
                'pre_acc': acc,
                'pre_f1': f1,
                'post_acc': acc2,
                'post_f1': f12,
                'retrain_time_s': retrain_time,
                'mem_diff_mb': mem_diff,
                'top_k_used': current_top_k,
                'features_used': len(current_features),
                'features': ','.join(current_features)
            })

        idx += 1

    # ---------------- SHAP summary ----------------
    try:
        import matplotlib.pyplot as plt
        sample_size = min(100, len(X_init), len(current_features))
        X_sample = X_init[current_features].sample(sample_size, random_state=42)
        fig = explain_model(clf, X_sample, model_type=model_name)
        if fig:
            os.makedirs("results", exist_ok=True)
            fig.savefig(f"results/{dataset}_{model_name}_shap_summary.png", bbox_inches='tight')
        plt.close('all')
    except Exception as e:
        print(f"⚠️ SHAP summary skipped: {e}")

    # ---------------- Save results ----------------
    if results:
        os.makedirs("results", exist_ok=True)
        results_df = pd.DataFrame(results)
        csv_path = f"results/{dataset}_{model_name}_runtime_summary.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\n📂 Saved runtime summary to {csv_path}")
    else:
        print("\n⚠️ No retraining events occurred, no summary saved.")

# ---------------- ENTRY ----------------
if __name__ == "__main__":
    nsl_files = [
        "data/NSL-KDD/KDDTrain+.txt",
        "data/NSL-KDD/KDDTest+.txt",
        "data/NSL-KDD/KDDTrain+_20Percent.txt"
    ]
    cicids_files = [
        "data/CICIDS/Monday-WorkingHours.pcap_ISCX.csv",
        "data/CICIDS/Tuesday-WorkingHours.pcap_ISCX.csv",
        "data/CICIDS/Wednesday-workingHours.pcap_ISCX.csv",
        "data/CICIDS/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "data/CICIDS/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "data/CICIDS/Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "data/CICIDS/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "data/CICIDS/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    ]

    # Test different adaptive methods
    top_k_methods = ["percentage", "threshold", "statistical"]
    
    for model_name in ["LogisticRegression", "RandomForest", "SVM", "XGBoost"]:
        for method in top_k_methods:
            print(f"\n=== Running {model_name} on NSL-KDD (method: {method}) ===")
            for file in nsl_files:
                run_pipeline(dataset="nsl", path=file, model_name=model_name,
                             init_size=3000, window_size=1000, top_k_method=method)

            print(f"\n=== Running {model_name} on CICIDS (method: {method}) ===")
            for file in cicids_files:
                run_pipeline(dataset="cicids", path=file, model_name=model_name,
                             init_size=3000, window_size=1000, top_k_method=method)

# import os
# import pandas as pd
# import numpy as np
# import time, psutil
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
# from load_data import load_and_combine_cicids, load_nsl_kdd
# from feature_analysis import preprocess
# from stream_generator import stratified_stream_windows
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from xgboost import XGBClassifier

# def get_model(model_name):
#     if model_name == "RandomForest":
#         return RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
#     elif model_name == "LogisticRegression":
#         return LogisticRegression(max_iter=1000, solver='lbfgs')
#     elif model_name == "SVM":
#         return SVC(probability=True)
#     elif model_name == "XGBoost":
#         return XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='logloss')
#     else:
#         raise ValueError(f"Unknown model: {model_name}")

# def validate_dataset_has_attacks(df, filename):
#     """Check if dataset has both normal and attack traffic"""
#     if 'label' in df.columns:
#         labels = df['label']
#     elif 'Label' in df.columns:
#         labels = df['Label']
#     else:
#         print(f"❌ No label column found in {filename}")
#         return False
    
#     unique_labels = labels.unique()
#     print(f"🔍 {filename}: Labels found = {unique_labels}, Distribution = {labels.value_counts().to_dict()}")
    
#     # For CICIDS: 0 = Normal, 1+ = Attacks
#     has_normal = 0 in unique_labels
#     has_attacks = len([x for x in unique_labels if x != 0]) > 0
    
#     if has_normal and has_attacks:
#         print(f"✅ Good: {filename} has both normal and attack traffic")
#         return True
#     else:
#         print(f"❌ Skip: {filename} has only {'normal' if has_normal else 'attack'} traffic")
#         return False

# def run_dataset_pipeline(dataset, file_list, model_name, init_size=1000, window_size=500):
#     print(f"\n{'='*60}")
#     print(f"=== FIXED VERSION: {model_name} on {dataset} ===")
#     print(f"{'='*60}")
    
#     # Load and VALIDATE data
#     if dataset == "cicids":
#         print("Loading CICIDS files (checking for attacks)...")
#         all_dfs = []
#         for file_path in file_list:
#             if os.path.exists(file_path):
#                 try:
#                     df_day = load_and_combine_cicids([file_path], nrows_per_file=2000)
#                     if df_day is not None and len(df_day) > 0:
#                         # ✅ CRITICAL: Only use files with both normal and attacks
#                         if validate_dataset_has_attacks(df_day, os.path.basename(file_path)):
#                             all_dfs.append(df_day)
#                         else:
#                             print(f"   Skipping {os.path.basename(file_path)} - insufficient class variety")
#                 except Exception as e:
#                     print(f"  ❌ Failed to load {file_path}: {e}")
        
#         if not all_dfs:
#             print("❌ No suitable CICIDS files found (need both normal and attack traffic)")
#             return
            
#         df = pd.concat(all_dfs, ignore_index=True)
        
#     elif dataset == "nsl":
#         print("Loading NSL-KDD files...")
#         dfs = []
#         for f in file_list:
#             if os.path.exists(f):
#                 try:
#                     df_file = load_nsl_kdd(f, nrows=2000)
#                     if df_file is not None and len(df_file) > 0:
#                         dfs.append(df_file)
#                         print(f"  {os.path.basename(f)}: {len(df_file)} samples")
#                 except Exception as e:
#                     print(f"  ❌ Failed to load {f}: {e}")
        
#         if not dfs:
#             print("❌ No NSL-KDD files loaded!")
#             return
            
#         df = pd.concat(dfs, ignore_index=True)
#         df = df.sample(frac=1, random_state=42).reset_index(drop=True)
#     else:
#         raise ValueError("dataset must be 'cicids' or 'nsl'")

#     print(f"\n📦 Final dataset: {df.shape}")
    
#     # Preprocess
#     try:
#         X, y = preprocess(df)
#         print(f"✅ Preprocessing successful")
#         print(f"🔢 Classes after preprocessing: {y.unique()}")
#         print(f"📊 Class distribution: {y.value_counts().to_dict()}")
#     except Exception as e:
#         print(f"❌ Preprocessing failed: {e}")
#         return

#     # 🚨 CRITICAL CHECK: Ensure we have at least 2 classes
#     if len(y.unique()) < 2:
#         print("❌❌❌ ERROR: Dataset has only one class after preprocessing!")
#         print("This is why you were getting 100% accuracy!")
#         print("Solution: Use different CICIDS files that contain attacks")
#         return

#     # Balance the dataset (but keep class variety)
#     class_counts = y.value_counts()
#     print(f"\n🎯 Original class distribution: {class_counts.to_dict()}")
    
#     # Take balanced samples from each class
#     n_per_class = min(500, min(class_counts))  # Adjust based on your dataset size
    
#     balanced_indices = []
#     for class_label in y.unique():
#         class_indices = y[y == class_label].sample(n_per_class, random_state=42).index
#         balanced_indices.extend(class_indices)
    
#     X_bal = X.loc[balanced_indices]
#     y_bal = y.loc[balanced_indices]
    
#     print(f"🔧 Balanced dataset: {X_bal.shape}")
#     print(f"📊 Balanced distribution: {y_bal.value_counts().to_dict()}")

#     # Time-aware split for CICIDS
#     if dataset == "cicids":
#         split_idx = int(0.6 * len(X_bal))
#         X_init = X_bal.iloc[:split_idx]
#         y_init = y_bal.iloc[:split_idx]
#         X_rest = X_bal.iloc[split_idx:]
#         y_rest = y_bal.iloc[split_idx:]
#     else:
#         X_init, X_rest, y_init, y_rest = train_test_split(
#             X_bal, y_bal, train_size=0.6, stratify=y_bal, random_state=42
#         )

#     print(f"\n📊 Training set: {X_init.shape} ({len(y_init.unique())} classes)")
#     print(f"📊 Testing set:  {X_rest.shape} ({len(y_rest.unique())} classes)")

#     # Train model
#     print(f"\n🎯 Training {model_name}...")
#     clf = get_model(model_name)
#     clf.fit(X_init, y_init)

#     # Test performance
#     train_preds = clf.predict(X_init)
#     train_acc = accuracy_score(y_init, train_preds)
    
#     test_preds = clf.predict(X_rest)
#     test_acc = accuracy_score(y_rest, test_preds)

#     print(f"📊 Training accuracy:  {train_acc:.4f}")
#     print(f"📊 Testing accuracy:   {test_acc:.4f}")
    
#     # 🎉 REALISTIC RESULTS CHECK
#     if 0.70 <= test_acc <= 0.95:
#         print("✅ EXCELLENT: Realistic accuracy achieved!")
#     elif test_acc > 0.95:
#         print("⚠️  High but plausible accuracy")
#     else:
#         print("📉 Low accuracy - model may need tuning")

#     # Streaming evaluation
#     print(f"\n🌊 Starting streaming evaluation...")
#     results = []
    
#     for idx, (X_win, y_win) in enumerate(stratified_stream_windows(X_rest, y_rest, window_size=200)):
#         if len(y_win.unique()) < 2:
#             continue
            
#         preds = clf.predict(X_win)
#         acc = accuracy_score(y_win, preds)
#         f1 = f1_score(y_win, preds, average='weighted', zero_division=0)
        
#         print(f"🔹 Window {idx}: acc={acc:.4f}, samples={len(y_win)}")
        
#         results.append({
#             "window": idx,
#             "model": model_name,
#             "dataset": dataset,
#             "accuracy": acc,
#             "f1": f1,
#             "samples": len(y_win)
#         })
        
#         if idx >= 10:  # Limit for testing
#             break

#     if results:
#         avg_acc = pd.DataFrame(results)['accuracy'].mean()
#         print(f"\n📈 Average streaming accuracy: {avg_acc:.4f}")
        
#         # Save results
#         os.makedirs("results", exist_ok=True)
#         results_df = pd.DataFrame(results)
#         results_df.to_csv(f"results/{dataset}_{model_name}_realistic_results.csv", index=False)
#         print(f"💾 Results saved to results/{dataset}_{model_name}_realistic_results.csv")

# # 🎯 USE THESE FILES - They contain attacks!
# if __name__ == "__main__":
#     # CICIDS files that CONTAIN ATTACKS:
#     cicids_files_with_attacks = [
#         "data/CICIDS/Wednesday-workingHours.pcap_ISCX.csv",                    # Has attacks
#         "data/CICIDS/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",  # Has attacks  
#         "data/CICIDS/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", # Has attacks
#         "data/CICIDS/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",    # Has attacks
#         "data/CICIDS/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"         # Has attacks
#     ]
    
#     nsl_files = [
#         "data/NSL-KDD/KDDTrain+.txt",
#         "data/NSL-KDD/KDDTest+.txt"
#     ]

#     print("🚀 RUNNING WITH ATTACK FILES - Should get realistic results now!")
    
#     # Test with one model first
#     run_dataset_pipeline("cicids", cicids_files_with_attacks, "RandomForest", 
#                        init_size=500, window_size=200)