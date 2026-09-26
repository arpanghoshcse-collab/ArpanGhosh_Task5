# 📊 Sales Prediction Using Machine Learning in Python

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning regression system that predicts product sales based on advertising expenditures across multiple media channels (**TV**, **Radio**, and **Newspaper**). 

The project compares a baseline **Multiple Linear Regression** model against non-linear architectures (**Polynomial Regression with Interaction Terms** and **Random Forest Regressor**), performs rigorous residual diagnostics, and delivers actionable marketing insights regarding channel return on investment (ROI) and cross-channel synergy.

---

## 📌 Executive Summary & Key Findings

1. **Top Impact Channel — TV Advertising**:
   * **Standardized Beta ($\beta^* = 0.737$)**: TV is the single largest driver of product sales.
   * **Random Forest Importance**: TV accounts for **62.6%** of total predictive importance.
   * **Marginal Return**: Every additional \$1,000 invested in TV generates **~45 to 53 units** of product sales.

2. **Secondary Driver with Multiplicative Synergy — Radio Advertising**:
   * **Standardized Beta ($\beta^* = 0.547$)**: Strong standalone performance (accounting for **36.2%** of Random Forest importance).
   * **Synergy Effect**: In Polynomial Regression, the **$TV \times Radio$ interaction coefficient is positive (+0.0011)**. Deploying TV (broad reach) and Radio (frequency/reminder) in tandem creates a super-linear multiplicative lift on sales.

3. **Negligible / Ineffective Channel — Newspaper Advertising**:
   * **Standardized Beta ($\beta^* = 0.011$)**: Near-zero effect when controlling for TV and Radio spend.
   * **Omitted Variable Confounding**: While simple bivariate correlation showed $r = 0.228$, this was an artifact of markets with high newspaper ad spend also having higher radio spend.
   * **Actionable Recommendation**: **Reallocate newspaper marketing budgets into TV and Radio channels.**

---

## 🏆 Model Performance Benchmark

Models were trained on **80% of the dataset (160 markets)** and evaluated on **20% unseen test data (40 markets)** using fixed `random_state=42`.

| Model Architecture | Train $R^2$ | Test $R^2$ | Test MAE | Test RMSE | 5-Fold CV $R^2$ (Mean $\pm$ Std) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | 0.8957 | 0.8994 | 1.4608 | 1.7816 | $0.8703 \pm 0.0718$ |
| **Polynomial Regression ($d=2$ with Interactions)** 🥇 | **0.9861** | **0.9869** | **0.5262** | **0.6426** | **$0.9778 \pm 0.0170$** |
| **Random Forest Regressor (100 Trees)** 🥈 | 0.9946 | 0.9799 | 0.6325 | 0.7962 | $0.9639 \pm 0.0122$ |

> **Key Insight**: The **Polynomial Regression model ($d=2$)** is the superior solution: it explains **98.7% of the variance** in unseen sales, reduces MAE by **64%** compared to the baseline, and produces homoscedastic, normally distributed residuals.

---

## 📁 Repository Structure

```text
sales_prediction/
├── data/
│   ├── raw/
│   │   └── Advertising.csv          # Canonical 200-market benchmark dataset
│   └── processed/
│       └── advertising_cleaned.csv  # Validated and cleaned dataset
├── models/
│   ├── linear_regression_baseline.joblib
│   ├── polynomial_regression_degree_2.joblib
│   ├── random_forest_regressor.joblib
│   └── best_model.joblib            # Production-ready serialized top model
├── notebooks/
│   └── sales_prediction.ipynb       # Fully executed, pre-rendered Jupyter Notebook
├── reports/
│   └── figures/
│       ├── eda_pairplot.png
│       ├── scatter_sales_vs_tv.png
│       ├── scatter_sales_vs_radio.png
│       ├── scatter_sales_vs_newspaper.png
│       ├── combined_scatters.png
│       ├── correlation_heatmap.png
│       ├── distribution_histograms.png
│       ├── model_comparison_metrics.png
│       ├── best_model_residuals.png
│       ├── baseline_residuals.png
│       ├── actual_vs_predicted.png
│       └── feature_importance_coefficients.png
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Data ingestion, cleaning & train-test split
│   ├── eda.py                       # Exploratory data analysis & figure generator
│   ├── models.py                    # Training, benchmarking, metrics & diagnostics
│   ├── interpret.py                 # Standardized coefficients & ROI analysis
│   ├── build_notebook.py            # Programmatic notebook builder
│   └── render_notebook.py           # In-process notebook renderer
├── requirements.txt                 # Pinned dependencies
└── README.md                        # Project documentation
```

---

## 🔬 Methodology & Workflow

### 1. Data Ingestion & Sanitization
* Loaded the 200-observation *Advertising* dataset.
* Removed redundant index column `Unnamed: 0`.
* Verified zero missing/null values and zero duplicate rows across all features.

### 2. Exploratory Data Analysis
* Computed descriptive statistics (mean, median, standard deviation, IQR, skewness, kurtosis).
* Analyzed bivariate relationships using pairplots and individual scatter plots with OLS regression lines.
* Generated a Pearson correlation matrix: TV ($r=0.782$), Radio ($r=0.576$), Newspaper ($r=0.228$).

### 3. Model Engineering & Benchmarking
* Implemented **Multiple Linear Regression** as an interpretable baseline ($R^2 = 0.899$).
* Developed **Polynomial Regression** with quadratic and interaction terms ($\text{TV} \times \text{Radio}$) to capture marketing synergies, achieving state-of-the-art accuracy ($R^2 = 0.987$).
* Trained a non-parametric **Random Forest Regressor** with 100 estimators ($R^2 = 0.980$).
* Evaluated models via MAE, RMSE, $R^2$, and 5-Fold Cross-Validation.

### 4. Residual Diagnostics
* Evaluated the best model for **homoscedasticity** and **random error distribution** via Residuals vs. Fitted values.
* Tested normality of errors with a **Normal Q-Q plot**, confirming Gaussian error distribution.
* Created parity plots of Actual vs. Predicted sales.

### 5. Interpretation & Channel Impact
* Standardized features to compute unit-free standardized beta coefficients ($\beta^*$).
* Extracted tree Gini impurity reduction from Random Forest.
* Concluded that **TV has the highest single impact**, followed by **Radio**, while **Newspaper spend has near-zero explanatory power**.

---

## 🚀 How to Run and Reproduce

### 1. Installation
Clone the repository and install requirements:
```bash
cd sales_prediction
pip install -r requirements.txt
```

### 2. Run Python Modules
Execute individual pipeline components from the project root:
```bash
# 1. Ingestion & Audit
python src/data_loader.py

# 2. Exploratory Data Analysis & Plots
python src/eda.py

# 3. Model Training & Diagnostics
python src/models.py

# 4. Feature Importance & Channel Impact
python src/interpret.py
```

### 3. Build & Render Jupyter Notebook
Generate and render the complete notebook with all code, tables, and inline charts:
```bash
python src/render_notebook.py
```
Open `notebooks/sales_prediction.ipynb` in VS Code, JupyterLab, or Google Colab. All execution outputs and visualizations are pre-rendered and immediately viewable.

---

## 📈 Marketing Simulation Example

Using the serialized top model (`models/best_model.joblib`), marketing planners can evaluate budget scenarios:

```python
import joblib
import pandas as pd

model = joblib.load("models/best_model.joblib")

# Scenario: $110k on TV, $40k on Radio, $0k on Newspaper
budget = pd.DataFrame([{"TV": 110.0, "Radio": 40.0, "Newspaper": 0.0}])
predicted_sales = model.predict(budget)[0]
print(f"Predicted Sales: {predicted_sales:.2f} thousand units")
# Output: Predicted Sales: 18.06 thousand units
```
