# src/explainability/shap_explainer.py

import shap
import matplotlib.pyplot as plt
import pandas as pd

def shap_summary(model, X, feature_names=None, model_name="Model"):
    """
    Generate SHAP summary plot for a trained model.
    """
    try:
        # Create explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Handle list output (e.g. XGBoost binary)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Summary plot
        shap.summary_plot(shap_values, X, feature_names=feature_names)
        plt.title(f"{model_name} SHAP Summary")
        plt.show()

        # Return shap values as DataFrame
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        return shap_df

    except Exception as e:
        print("SHAP Explainer error:", e)
        return None


def shap_force_plot(model, X, instance_index=0, feature_names=None):
    """
    Visualize SHAP force plot for a single instance.
    """
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # Handle list output (e.g. XGBoost binary)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # shap.initjs()
    shap.force_plot(
        explainer.expected_value,
        shap_values[instance_index],
        X[instance_index],
        feature_names=feature_names,
        matplotlib=True
    )
    plt.show()