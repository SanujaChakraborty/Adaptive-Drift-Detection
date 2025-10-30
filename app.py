import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set page title
st.set_page_config(page_title="Drift Detection Dashboard", layout="wide")

st.title("🔍 Data Drift Monitoring Dashboard")

# Debug info (optional)
# st.write("Current working directory:", os.getcwd())
# st.write("Files in results/:", os.listdir("results"))

# File paths
nsl_path = os.path.join("results", "nsl_results.csv")
cicids_path = os.path.join("results", "cicids_results.csv")

# Check files exist
if not os.path.exists(nsl_path) or not os.path.exists(cicids_path):
    st.error("❌ One or more result files not found in the 'results' folder.")
else:
    # Load data
    nsl = pd.read_csv(nsl_path)
    cic = pd.read_csv(cicids_path)

    st.success("✅ Successfully loaded both NSL-KDD and CICIDS results.")

    # Select dataset
    dataset_choice = st.selectbox("Select Dataset:", ["NSL-KDD", "CICIDS"])

    if dataset_choice == "NSL-KDD":
        sub = nsl
    else:
        sub = cic

    st.dataframe(sub.head())

    # Plot results
    st.subheader("📈 Model Performance over Windows")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sub['window'], sub['pre_acc'], label='Pre-Retrain Accuracy', marker='o')
    ax.plot(sub['window'], sub['post_acc'], label='Post-Retrain Accuracy', marker='s')
    ax.set_xlabel("Window")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{dataset_choice} Accuracy Trend")
    ax.legend()
    st.pyplot(fig)

    # F1 Score plot
    st.subheader("🎯 F1 Score Trend")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(sub['window'], sub['pre_f1'], label='Pre-Retrain F1', marker='o')
    ax2.plot(sub['window'], sub['post_f1'], label='Post-Retrain F1', marker='s')
    ax2.set_xlabel("Window")
    ax2.set_ylabel("F1 Score")
    ax2.set_title(f"{dataset_choice} F1 Score Trend")
    ax2.legend()
    st.pyplot(fig2)

    # Retraining time and memory
    st.subheader("⚙️ Retraining Efficiency Metrics")
    col1, col2 = st.columns(2)
    with col1:
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        ax3.plot(sub['window'], sub['retrain_time_s'], color='purple', marker='d')
        ax3.set_xlabel("Window")
        ax3.set_ylabel("Retraining Time (s)")
        ax3.set_title("Retraining Time per Window")
        st.pyplot(fig3)

    with col2:
        fig4, ax4 = plt.subplots(figsize=(5, 3))
        ax4.plot(sub['window'], sub['mem_diff_mb'], color='green', marker='d')
        ax4.set_xlabel("Window")
        ax4.set_ylabel("Memory Diff (MB)")
        ax4.set_title("Memory Usage Change per Window")
        st.pyplot(fig4)

    # Summary metrics
    st.subheader("📊 Summary Statistics")
    st.write(sub.describe())
