# Data Scientist Salary Analysis & Prediction Across Continents

## Project Overview
This project explores global data scientist salary patterns across five continents using a real-world dataset. Starting from ~52,000 salary records, the dataset was filtered to Data Scientist roles only, resulting in 5,873 records. It combines exploratory data analysis (EDA), statistical testing, and predictive modelling to understand what drives salary differences — and honestly confronts the limitations of imbalanced data.

---

## Table of Contents
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)
- [Modelling Results](#modelling-results)
- [Limitations](#limitations)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Recommendations & Future Work](#recommendations--future-work)

---

## Dataset
- **Source:** `salaries.csv`
- **Original records:** ~52,000
- **Filtered records:** 5,873 (Data Scientist roles only, after deduplication)
- **Key columns:** `experience_level`, `employment_type`, `salary_in_usd`, `employee_residence`, `company_location`, `company_size`
- **Continent mapping:** Country codes manually mapped to 5 continents (Africa, Americas, Asia, Europe, Oceania)

### Data Imbalance Warning
| Continent | Records | Percentage |
|-----------|---------|------------|
| Americas  | 5,227   | 89.00%     |
| Europe    | 529     | 9.01%      |
| Asia      | 53      | 0.90%      |
| Oceania   | 43      | 0.73%      |
| Africa    | 21      | 0.36%      |

---

## Project Structure
```
SalaryPrediction.ipynb
│
├── 1. Data Cleaning & Interrogation
│   ├── Null checks
│   ├── Duplicate removal
│   └── Filtering to Data Scientist roles
│
├── 2. Exploratory Data Analysis
│   ├── Geographical & Distributional Questions
│   │   ├── Mean & median salary by continent
│   │   ├── Salary variance & outliers (boxplots)
│   │   └── Data representation by continent
│   ├── Experience & Role Demographics
│   │   ├── Salary by experience level & continent
│   │   ├── Senior Asia/Africa vs Entry Americas/Europe
│   │   └── Non-linear experience-salary relationships
│   └── Employment Structure & Macro Factors
│       ├── Foreign vs local company proportions (Africa & Asia)
│       ├── Salary comparison: foreign vs local firms
│       └── Company size vs salary by continent
│
├── 3. Statistical Testing
│   └── One-way ANOVA (continent vs salary)
│
├── 4. Modelling
│   ├── Feature encoding (pd.get_dummies)
│   ├── Train/test split (80/20)
│   ├── Feature scaling (StandardScaler)
│   ├── Linear Regression (unbalanced & balanced)
│   ├── Random Forest (default, tuned via Grid Search, balanced)
│   └── Feature Importance comparison
│
└── 5. Conclusion & Recommendations
```

---

## Key Findings

- **Americas dominates** — highest salaries at every experience level and company size, consistently
- **Experience level is a universal predictor** — EN → MI → SE → EX salary progression holds across all continents
- **The gap explodes at Executive level** — Oceania executives earn ~$330k median, Americas ~$195k, while Africa's EX data is too sparse to be reliable
- **Asia's local market is the worst paying** — Asian data scientists working for foreign companies earn noticeably more than those working locally
- **Africa and Asia's results should be treated with caution** — sample sizes are too small for reliable conclusions
- **Non-linear relationships exist** — Africa shows a V-shape salary dip at mid-level; most continents are non-linear

---

## Modelling Results

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression (Unbalanced) | 48,030 | 64,960 | 0.1973 |
| Random Forest (Unbalanced) | 48,194 | 65,225 | 0.1907 |
| Random Forest Tuned (Unbalanced) | 48,024 | 64,947 | 0.1976 |
| Linear Regression (Balanced) | 44,604 | 56,753 | 0.2576 |
| Random Forest (Balanced) | 44,558 | 56,808 | 0.2561 |

**Best model:** Linear Regression (Balanced) — R² = 0.2576

---

## Limitations

- Americas accounts for ~89% of records after filtering, making the model heavily biased toward Americas salary patterns
- Africa (21 rows) and Asia (53 rows) are too underrepresented for reliable predictions
- Available features (continent, experience level, company size, employment type) have limited predictive power alone
- Key salary drivers like job title, specific skills, and cost of living are absent from the dataset
- Feature importance was virtually unchanged after balancing, confirming the imbalance problem runs deeper than simple undersampling

---

## Technologies Used

```
Python 3.13
pandas
numpy
plotnine (ggplot2-style visualisation)
scikit-learn (LinearRegression, RandomForestRegressor, GridSearchCV, StandardScaler)
scipy (f_oneway — ANOVA)
janitor
matplotlib
```

---

## How to Run

1. Clone the repository
```bash
git clone https://github.com/yourusername/salary-prediction.git
cd salary-prediction
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

3. Install dependencies
```bash
pip install pandas numpy plotnine scikit-learn scipy pyjanitor matplotlib
```

4. Launch Jupyter
```bash
jupyter notebook SalaryPrediction.ipynb
```

---

## Recommendations & Future Work

1. **Address Data Imbalance** — Collect more data from Africa and Asia; apply SMOTE on Asia once sufficient data is available (minimum 500–1000 rows recommended)
2. **Feature Enrichment** — Merge cost of living indices (Numbeo, World Bank PPP), add job title, skills, and years of experience as a continuous variable
3. **Modelling Improvements** — Explore XGBoost/LightGBM; train separate models per continent; log-transform salary before modelling to handle right skew
4. **Statistical Robustness** — Run two-way ANOVA on balanced data; apply Tukey post-hoc tests per continent pair to identify specific regional differences
5. **Business Note** — Salary benchmarks for Africa and Asia should be treated with very low confidence until more representative data is collected; Americas salary patterns should not be used as a global benchmark

---

## Author
QUEEN ABICHE
