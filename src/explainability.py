import shap
import matplotlib.pyplot as plt

def explain_model(model, X_sample, model_type=None):
    """
    Generate SHAP summary plot.
    Handles both tree-based and linear/non-tree models.
    Returns the figure handle (not shown directly).
    """
    plt.close('all')

    # Choose explainer
    if model_type in ["RandomForest", "XGBoost"]:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X_sample, feature_dependence="independent")

    shap_values = explainer.shap_values(X_sample)

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    return fig
