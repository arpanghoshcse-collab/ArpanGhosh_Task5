"""
Exploratory Data Analysis (EDA) module for Sales Prediction.
Computes descriptive statistics and produces publication-quality visualizations:
- Pairplot of all variables
- Individual scatter plots with regression trendlines (Sales vs TV, Radio, Newspaper)
- Combined multi-panel scatter plot
- Pearson correlation matrix heatmap
- Distribution histograms and boxplots
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Configure matplotlib temporary cache to eliminate warnings
os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.data_loader import load_cleaned_data

FIGURES_DIR = BASE_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Visual styling
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


def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute comprehensive descriptive statistics."""
    desc = df.describe().T
    desc["median"] = df.median()
    desc["skew"] = df.skew()
    desc["kurtosis"] = df.kurtosis()
    desc["IQR"] = desc["75%"] - desc["25%"]
    return desc


def plot_pairplot(df: pd.DataFrame, save_path=FIGURES_DIR / "eda_pairplot.png"):
    """Plot pairplot showing bivariate relationships and marginal distributions."""
    g = sns.pairplot(
        df,
        diag_kind="kde",
        plot_kws={"alpha": 0.7, "color": "#1f77b4", "s": 35, "edgecolor": "k", "linewidths": 0.5},
        diag_kws={"fill": True, "color": "#2ca02c", "linewidth": 1.5}
    )
    g.fig.suptitle("Pairplot of Advertising Features and Sales", y=1.02, fontsize=14, fontweight="bold")
    g.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_individual_scatters(df: pd.DataFrame, figures_dir=FIGURES_DIR):
    """
    Generate individual scatter plots for:
    - Sales vs. TV spend
    - Sales vs. Radio spend
    - Sales vs. Newspaper spend
    """
    channels = [
        ("TV", "TV Spend ($1,000s)", "#1f77b4"),
        ("Radio", "Radio Spend ($1,000s)", "#ff7f0e"),
        ("Newspaper", "Newspaper Spend ($1,000s)", "#2ca02c")
    ]

    for col, label, color in channels:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.regplot(
            data=df,
            x=col,
            y="Sales",
            ax=ax,
            color=color,
            scatter_kws={"alpha": 0.75, "s": 45, "edgecolor": "black", "linewidths": 0.5},
            line_kws={"color": "darkred", "linewidth": 2, "label": "OLS Trendline"}
        )
        r = df[col].corr(df["Sales"])
        ax.set_title(f"Product Sales vs. {col} Advertising Spend\n(Pearson r = {r:.3f})", fontweight="bold")
        ax.set_xlabel(label)
        ax.set_ylabel("Sales (1,000 units)")
        ax.legend(frameon=True, facecolor="white", loc="upper left")
        plt.tight_layout()
        save_file = figures_dir / f"scatter_sales_vs_{col.lower()}.png"
        fig.savefig(save_file, dpi=300)
        plt.close(fig)
        print(f"Saved: {save_file}")


def plot_combined_scatters(df: pd.DataFrame, save_path=FIGURES_DIR / "combined_scatters.png"):
    """Generate a combined 3-panel scatter plot comparing all channels."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    channels = [
        ("TV", "TV Spend ($1,000s)", "#1f77b4"),
        ("Radio", "Radio Spend ($1,000s)", "#ff7f0e"),
        ("Newspaper", "Newspaper Spend ($1,000s)", "#2ca02c")
    ]

    for ax, (col, label, color) in zip(axes, channels):
        r = df[col].corr(df["Sales"])
        sns.regplot(
            data=df,
            x=col,
            y="Sales",
            ax=ax,
            color=color,
            scatter_kws={"alpha": 0.75, "s": 40, "edgecolor": "k", "linewidths": 0.5},
            line_kws={"color": "red", "linewidth": 2}
        )
        ax.set_title(f"Sales vs. {col}\n(r = {r:.3f})", fontweight="bold")
        ax.set_xlabel(label)
        if ax == axes[0]:
            ax.set_ylabel("Sales (1,000 units)")
        else:
            ax.set_ylabel("")

    fig.suptitle("Impact of Media Advertising Expenditures on Product Sales", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_correlation_heatmap(df: pd.DataFrame, save_path=FIGURES_DIR / "correlation_heatmap.png"):
    """Plot annotated correlation matrix heatmap."""
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
        ax=ax
    )
    ax.set_title("Correlation Matrix Heatmap: Spend vs. Sales", fontweight="bold", pad=15)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_feature_distributions(df: pd.DataFrame, save_path=FIGURES_DIR / "distribution_histograms.png"):
    """Plot histograms and KDEs for all 4 variables."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    cols = ["TV", "Radio", "Newspaper", "Sales"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for ax, col, color in zip(axes.flatten(), cols, colors):
        sns.histplot(df[col], kde=True, ax=ax, color=color, bins=15, alpha=0.6)
        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.1f}")
        ax.axvline(median_val, color="black", linestyle=":", linewidth=1.5, label=f"Median: {median_val:.1f}")
        ax.set_title(f"Distribution of {col}", fontweight="bold")
        ax.set_xlabel(f"{col} Spend / Value")
        ax.legend(frameon=True, facecolor="white")

    plt.suptitle("Feature & Target Distributions with KDEs", fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")


def run_eda():
    """Execute complete EDA pipeline and output summary statistics."""
    df = load_cleaned_data()
    print("\n--- Summary Descriptive Statistics ---")
    stats = compute_summary_statistics(df)
    print(stats.round(3))

    print("\n--- Correlation with Sales ---")
    print(df.corr()["Sales"].sort_values(ascending=False).round(4))

    print("\n--- Generating Visualizations ---")
    plot_pairplot(df)
    plot_individual_scatters(df)
    plot_combined_scatters(df)
    plot_correlation_heatmap(df)
    plot_feature_distributions(df)
    print("EDA completed successfully.")


if __name__ == "__main__":
    run_eda()
