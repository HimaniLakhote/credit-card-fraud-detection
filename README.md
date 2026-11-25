# 🛡️ CreditGuard: Credit Card Fraud Detection

Machine learning course project (Project 19). It detects fraudulent credit card transactions in a highly imbalanced dataset and includes a live Streamlit dashboard.

**Live demo:** PASTE_YOUR_STREAMLIT_LINK_HERE

## Problem
Only about 0.17% of transactions are fraud (492 of 284,807). A model that always says "not fraud" would be 99.8% accurate but useless, so we need methods and metrics that handle imbalance.

## Dataset
[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): European card transactions over two days. Features are `Time`, `V1`-`V28` (anonymised PCA components), `Amount`, and the label `Class` (1 = fraud). The raw file is not included in this repo (144 MB).

## Approach
- Stratified 80/20 train/test split (`random_state=42`) so both sets keep the same fraud rate.
- **Class weighting** (`class_weight`, and `scale_pos_weight` for XGBoost) so the models pay more attention to rare frauds.
- Five models compared: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM.
- **PR-AUC is the main metric**, because accuracy is misleading on imbalanced data. We also report precision, recall, F1 and ROC-AUC.

## Results (test set, 98 frauds)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Train time (s) |
|---|---|---|---|---|---|---|---|
| **LightGBM** | 0.9993 | 0.783 | 0.847 | 0.814 | 0.965 | **0.864** | 6.1 |
| Random Forest | 0.9993 | 0.766 | 0.837 | 0.800 | 0.981 | 0.801 | 84.0 |
| XGBoost | 0.9968 | 0.331 | 0.867 | 0.479 | 0.977 | 0.793 | 3.3 |
| Logistic Regression | 0.9745 | 0.059 | 0.918 | 0.110 | 0.971 | 0.718 | 2.3 |
| Decision Tree | 0.9806 | 0.071 | 0.847 | 0.131 | 0.918 | 0.471 | 25.7 |

**Best model: LightGBM** (highest PR-AUC, with the best balance of precision and recall). Logistic Regression catches the most fraud but raises many false alarms (precision 0.059).

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dashboard
- **Model results:** comparison table and charts.
- **Live monitor:** replays held-out transactions in real time with a fraud-threshold slider.
- **Upload CSV:** score your own file and download the flagged rows.

## Repo layout
`app.py`, `requirements.txt`, notebook (`CreditGuard_Fraud_Detection_Final.ipynb`), `models/fraud_detection_model.pkl`, `model_comparison.csv`, `feature_importance.csv`, `demo_transactions.csv`.

## Limitations
Educational demo only. Results come from a single train/test split with few fraud cases, so they have some variance. The features are anonymised, and real systems need monitoring, retraining and human review.
