# Explainable and Adaptive Intrusion Detection System under Data Drift with Selective Labeling and Feature Stability

# Project Overview
This project implements an Adaptive Concept Drift Detection and Retraining Framework for Intrusion Detection Systems (IDS) using the NSL-KDD and CICIDS2017 datasets.
It continuously monitors model performance on streaming data, detects data drift using ADWIN, and selectively retrains models to maintain accuracy over time.

# Objectives
```
    Detect data drift in streaming network traffic using ADWIN (Adaptive Windowing)
    Perform feature stability analysis to identify changing network attributes
    Implement selective retraining of the model instead of full retraining
    Explain predictions using SHAP (SHapley Additive exPlanations)
    Deploy results as an interactive Streamlit dashboard
```
# Methodology
```
    Data Preprocessing
    Handled missing values, label encoding, scaling
    Split datasets into windowed batches (Window 0–15) to simulate streaming data
```
# Model Training
```
    Initial training on Window 0
    Drift detection using ADWIN on subsequent windows
```
# Selective Retraining
```
    Retrains only when drift is detected
    Updates model incrementally to save time and resources
```
# Explainability (SHAP)
```
    Visualizes feature importance per window
    Highlights key attack features affecting decisions
```
# Dashboard (Streamlit)
Displays accuracy, F1-score, drift points, and retraining metrics

# Datasets
```
    NSL-KDD – Network traffic-based intrusion detection dataset
    CICIDS2017 – Modern IDS dataset simulating realistic attack scenarios
    Each dataset was divided into 16 windows (Window 0–15) to emulate streaming data flow.
```
# Tools & Libraries
```
    Python, Pandas, NumPy, Scikit-learn, XGBoost
    ADWIN (from river library) for drift detection
    SHAP for explainability
    Streamlit for dashboard visualization
    Matplotlib / Seaborn for plots
```

# Dashboard Features
```
    Accuracy & F1 trends (before/after retraining)
    Drift detection visualization
    SHAP-based feature importance
    Retraining time and memory efficiency metrics
```               

# Run locally:
streamlit run app.py

# Project Structure
```
LTI_PROJECT/
├── data/
├── notebooks/
├── results/        # Output CSVs, plots, SHAP summaries
├── src/            # Source scripts (data preprocessing, modeling)
├── venv/           # Virtual environment
├── .gitignore
├── app.py          # Streamlit dashboard
├── README.md       # Project documentation
└── requirements.txt
```

# Key Findings
```
    Drift detection successfully triggered when statistical shifts occurred in network data.
    Selective retraining reduced total training time by ~60%.
    SHAP revealed “src_bytes”, “dst_bytes”, and “flag” as most influential features in NSL-KDD.
```


# Future Work
```
    Extend to real-time network stream ingestion
    Integrate multimedia packet analysis (IP-based flows)
    Deploy on Flask API or cloud (AWS Lambda) for continuous monitoring
```


# Author
Sanuja Chakraborty 
M.Tech, Computer Science and Engineering (AI & DS)


