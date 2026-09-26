"""
Model Training, Evaluation, and Diagnostic module for Sales Prediction.
Trains:
1. Linear Regression (Baseline)
2. Polynomial Regression (degree=2 with interaction terms)
3. Random Forest Regressor
Evaluates:
- MAE, RMSE, R2, and 5-Fold Cross-Validation
- Residual diagnostics (homoscedasticity, normality, independence)
- Model serialization with joblib
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold

from src.data_loader import get_train_test_data

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = BASE_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train Linear Regression, Polynomial Regression, and Random Forest Regressor.
    Returns dictionary of results and fitted model objects.
    """
    models = {
        "Linear Regression (Baseline)": LinearRegression(),
        "Polynomial Regression (Degree 2)": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("linear", LinearRegression())
        ]),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, random_state=42, max_depth=6
        )
    }

    results = []
    trained_models = {}
    predictions = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        # Fit model
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Predict
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        predictions[name] = y_test_pred

        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # Cross Validation R2
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        results.append({
            "Model": name,
            "Train R²": train_r2,
            "Test R²": test_r2,
            "MAE": mae,
            "RMSE": rmse,
            "5-Fold CV R² (Mean)": cv_mean,
            "5-Fold CV R² (Std)": cv_std
        })

        # Save model artifact
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        model_path = MODELS_DIR / f"{safe_name}.joblib"
        joblib.dump(model, model_path)
        print(f"Serialized model: {model_path}")

    results_df = pd.DataFrame(results)
    
    # Identify best model based on Test R2
    best_row = results_df.sort_values(by="Test R²", ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_model = trained_models[best_model_name]
    
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    print(f"\nBest Model: '{best_model_name}' saved to models/best_model.joblib")

    return results_df, trained_models, predictions, best_model_name


def plot_model_comparison(results_df: pd.DataFrame, save_path=FIGURES_DIR / "model_comparison_metrics.png"):
    """Plot comparative metrics across all models."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [
        ("Test R²", "R² Score (Higher is Better)", "#2ca02c", (0.8, 1.0)),
        ("MAE", "Mean Absolute Error (Lower is Better)", "#1f77b4", None),
        ("RMSE", "Root Mean Squared Error (Lower is Better)", "#d62728", None)
    ]

    for ax, (col, title, color, ylim) in zip(axes, metrics):
        sns.barplot(
            data=results_df,
            x="Model",
            y=col,
            ax=ax,
            hue="Model",
            palette=[color] * len(results_df),
            legend=False,
            edgecolor="black"
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(col)
        ax.tick_params(axis='x', rotation=15)
        if ylim:
            ax.set_ylim(ylim)
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f"{height:.3f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        xytext=(0, 3), textcoords='offset points',
                        fontweight="bold", fontsize=10)

    plt.suptitle("Regression Benchmark Comparison across Media Spending Models", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_residuals_diagnostic(y_test, y_pred, model_name, save_path=FIGURES_DIR / "best_model_residuals.png"):
    """
    Residual plot for the best model:
    - Residuals vs. Predicted Values (homoscedasticity and randomness check)
    - Residual Distribution & Q-Q Plot (normality check)
    """
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.8, color="#1f77b4", edgecolor="black", s=50, label="Residuals ($y - \\hat{y}$)")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=2, label="Zero Error Line")
    
    # Visual LOWESS / polynomial fit to test whether errors are truly random or show curvature
    sns.regplot(
        x=y_pred,
        y=residuals,
        scatter=False,
        ax=axes[0],
        color="orange",
        line_kws={"linestyle": ":", "linewidth": 2, "label": "Residual Trendline"}
    )
    axes[0].set_title(f"Residuals vs. Predicted Sales ({model_name})\nCheck for Random Distribution vs. Systematic Error", fontweight="bold")
    axes[0].set_xlabel("Predicted Sales (1,000 units)")
    axes[0].set_ylabel("Residuals ($y - \\hat{y}$)")
    axes[0].legend(frameon=True, facecolor="white", loc="best")

    # 2. Q-Q Plot for Normality
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].get_lines()[0].set_markerfacecolor('#2ca02c')
    axes[1].get_lines()[0].set_markeredgecolor('black')
    axes[1].get_lines()[0].set_alpha(0.8)
    axes[1].get_lines()[1].set_color('red')
    axes[1].get_lines()[1].set_linewidth(2)
    axes[1].set_title("Normal Q-Q Plot of Residuals\nCheck for Normal Distribution of Errors", fontweight="bold")
    axes[1].set_xlabel("Theoretical Quantiles")
    axes[1].set_ylabel("Sample Quantiles")

    plt.suptitle(f"Residual Diagnostic Analysis: {model_name}", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_actual_vs_predicted(y_test, y_pred, model_name, save_path=FIGURES_DIR / "actual_vs_predicted.png"):
    """Plot actual vs predicted sales with perfect prediction diagonal line."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, color="#2b5c8f", alpha=0.8, edgecolors="k", s=50, label="Predictions")
    
    min_val = min(y_test.min(), y_pred.min()) - 1
    max_val = max(y_test.max(), y_pred.max()) + 1
    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Perfect Agreement (y = ŷ)")
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    ax.set_title(f"Actual vs. Predicted Sales: {model_name}\n($R^2 = {r2:.3f}$, MAE = {mae:.3f})", fontweight="bold")
    ax.set_xlabel("Actual Sales (1,000 units)")
    ax.set_ylabel("Predicted Sales (1,000 units)")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")


def run_modeling_pipeline():
    """Run full modeling and evaluation pipeline."""
    X_train, X_test, y_train, y_test = get_train_test_data()

    print("\n--- Training and Benchmarking Models ---")
    results_df, trained_models, predictions, best_name = train_and_evaluate_models(
        X_train, X_test, y_train, y_test
    )

    print("\n--- Model Benchmark Results ---")
    print(results_df.to_string(index=False))

    print("\n--- Generating Diagnostic Figures ---")
    plot_model_comparison(results_df)

    best_pred = predictions[best_name]
    plot_residuals_diagnostic(y_test, best_pred, best_name)
    plot_actual_vs_predicted(y_test, best_pred, best_name)

    # Also generate baseline linear regression residual plot for comparison
    if "Linear Regression (Baseline)" != best_name:
        plot_residuals_diagnostic(
            y_test,
            predictions["Linear Regression (Baseline)"],
            "Linear Regression (Baseline)",
            save_path=FIGURES_DIR / "baseline_residuals.png"
        )

    print("Modeling and diagnostic pipeline completed successfully.")
    return results_df, trained_models, predictions, best_name


if __name__ == "__main__":
    run_modeling_pipeline()
