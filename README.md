# Towards Intelligent Roadway-Policy Making: A Regression Analysis

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive regression analysis to predict road traffic fatalities at the provincial level in Iran. This project evaluates 13 different regression models to build a robust framework for data-driven road safety policy, ultimately selecting **Lasso** and **ElasticNet** as the most stable, accurate, and interpretable models.

## 📊 Project Overview

The primary goal of this research is to develop a statistical model to predict and understand the key factors influencing the number of deaths from road accidents across 31 provinces of Iran, using aggregated annual data from 1395-1402 (approx. 2016-2023).

The project leverages a multi-source dataset including:
*   **Traffic Data**: Vehicle counts by class, traffic violations, and average speed from Iran's Road Maintenance and Transportation Organization (System 141).
*   **Weather Data**: Annual average temperature and precipitation.
*   **Fatality Data**: Official statistics on road accident deaths from the Forensic Medicine Organization.

After data cleaning and preprocessing, the final analysis was conducted on a dataset of **244 observations** (province-years).

### Key Research Questions
- Which traffic, vehicle, and environmental factors are the strongest predictors of road fatalities?
- Can we quantify the impact of speeding violations on the number of deaths?
- How does fatality risk vary between different provinces, even after controlling for traffic volume?
- Which regression models provide the best balance of predictive accuracy and interpretability for this problem?

---

## 🎯 Key Findings

The **ElasticNet** and **Lasso** models produced highly stable and interpretable results. The following are the most significant factors identified. Coefficients represent the estimated change in the number of annual deaths for a one-standard-deviation increase in the feature.

### Significant Factors Influencing Fatalities

| Factor | Model | Coefficient | p-value | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Speeding Violations (TV_S)** | ElasticNet | **+31.80** | 0.013 | **The most critical factor.** A one standard deviation increase in speeding violations is associated with approximately **32 additional deaths** per year in a province. |
| **Province: Fars** | ElasticNet | **+643.42** | < 0.001 | Fars province has the highest baseline risk, with ~643 more deaths annually than the reference province, after accounting for all other factors. |
| **Province: Sistan & Baluchestan** | ElasticNet | **+473.97** | < 0.001 | Shows an extremely high underlying risk compared to other regions. |
| **Year: 1402 (2023-24)** | ElasticNet | **+113.96** | < 0.001 | Indicates a significant post-COVID rebound in fatality risk, with a baseline increase of ~114 deaths. |
| **Year: 1401 (2022-23)** | ElasticNet | **+105.61** | < 0.001 | The increasing trend in risk is also highly significant for this year. |
| **Vehicle Class 2 (Trucks/Minibuses)** | ElasticNet | **-43.60** | 0.026 | **Protective effect.** A higher volume of Class 2 vehicles is associated with a *decrease* in fatalities, possibly due to a traffic "pacing effect." |

**Key Insight**: The analysis reveals that specific behavioral issues (speeding) and regional factors (province-specific risks) are far more predictive of fatalities than aggregate metrics like average speed or weather conditions, which were found to be statistically insignificant.

---

## 🏆 Model Performance

After an exhaustive evaluation of 13 models, **Lasso** and **ElasticNet** were selected for their superior performance, stability, and interpretability. Their performance metrics were nearly identical and top-of-the-class.

### Final Model Performance (ElasticNet & Lasso)

| Metric | Test Set Value | Cross-Validation Value | Interpretation |
| :--- | :--- | :--- | :--- |
| **R² Score** | **0.94** | **0.89 ± 0.05** | The models explain **94%** of the variance in road fatalities on unseen test data, demonstrating excellent predictive power. |
| **RMSE** | **~49 deaths** | **~72 deaths** | On average, the model's predictions on the test set were off by only 49 fatalities. |
| **R² (Bootstrap 95% CI)** | [0.89, 0.97] | - | The model's R² is robustly high, with the 95% confidence interval confirming strong performance. |
| **R² (Nested CV, 5 Seeds)** | - | **~0.90 ± 0.03** | **Exceptional stability.** The model's performance is consistent across different data splits, highlighting its reliability. |

These results, taken directly from **Tables 3 and 4 (page 6)** of the report, confirm that the selected models are highly effective and reliable for this prediction task.

---

## 🛠️ Technical Stack & Methodology

### Core Technologies
```python
# Modeling
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# ...and others including CatBoost, XGBoost, SVR.

# Data Processing & Evaluation
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
```

### Key Methodological Features

1.  **Data Aggregation**: Daily, road-level traffic data was aggregated to the annual, provincial level to match the granularity of the target variable (fatalities).
2.  **Multicollinearity Management**: The Variance Inflation Factor (VIF) was used to diagnose and remove features with high multicollinearity (`num_c1` was removed as its VIF was > 40).
3.  **Robust Model Selection**: A two-stage filtering process was employed:
    *   **Stage 1**: Models were evaluated using **Repeated Nested Cross-Validation** to ensure stable and unbiased performance estimates. Only models with high and consistent scores (mean R² - std R² ≥ 0.4) were kept.
    *   **Stage 2**: The remaining models were analyzed for bias-variance trade-offs and overfitting using learning curves.
4.  **Final Evaluation**: The top models were assessed using Bootstrap 95% Confidence Intervals and further Nested CV runs to confirm stability and quantify uncertainty.

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn
```

### Running the Analysis
The project is contained within a Jupyter Notebook that performs the end-to-end analysis:
1.  **Data Loading and Preprocessing**: Cleans data, handles missing values, and aggregates features.
2.  **Exploratory Data Analysis (EDA)**: Visualizes relationships between features and the target.
3.  **Multicollinearity Check**: Calculates VIF and removes problematic features.
4.  **Model Training and Evaluation**: Implements the rigorous two-stage model selection strategy.
5.  **Results Interpretation**: Analyzes the coefficients of the final models (Lasso and ElasticNet) to derive policy insights.

---

## 📚 Data Sources

-   **Traffic and Vehicle Data**: [Iran Road Maintenance and Transportation Organization (RMTO), System 141](http://www.141.ir)
-   **Fatality Statistics**: [Iranian Legal Medicine Organization (LMO)](http://lmo.ir)
-   **Weather Data**: [World Bank Climate Change Knowledge Portal](https://climateknowledgeportal.worldbank.org/) and [Time and Date](https://www.timeanddate.com/)

---

## 👥 Authors

| Name | Student ID |
| :--- | :--- |
| Arshia Dehghan | 401100382 |
| Abolfazl Moslemi | 401100506 |
| Ali Mohammad-Zadeh Shabestari | 401106482 |

**Institution**: Sharif University of Technology  
**Course**: Regression Analysis (Project 1) - April 2025 (فروردین ۱۴۰۴)

---

## 📄 Full Report

For a complete description of the methodology, theoretical framework, and detailed results, please see the full [`Report.pdf`](Report.pdf) (written in Persian).
