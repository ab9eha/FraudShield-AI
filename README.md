
# FraudShield AI 🛡️

### Credit Card Fraud Detection Using Machine Learning

FraudShield AI is an educational financial cybersecurity project that uses machine learning to identify potentially fraudulent credit card transactions. It provides an interactive Streamlit dashboard for dataset exploration, model evaluation, fraud probability analysis, and prediction review.

## Features

- Credit card transaction dataset analysis
- Random Forest classification
- Class imbalance handling with class-weighted learning
- Precision, recall, F1-score, ROC-AUC, and PR-AUC
- Confusion matrix and feature importance visualizations
- Adjustable fraud classification threshold
- Test-set prediction explorer and CSV export
- Synthetic demo mode for testing the interface

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Plotly

## Dataset

The project supports the public Kaggle Credit Card Fraud Detection dataset.

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download `creditcard.csv` separately. Do not commit the dataset to GitHub.

## Run Locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Methodology

1. Validate and preprocess transaction data.
2. Separate features from the `Class` target.
3. Split data into stratified training and test sets.
4. Train a Random Forest classifier using class weighting.
5. Evaluate performance with metrics suited to rare fraud events.
6. Visualize predictions and feature importance.

## Limitations

This is an educational prototype, not a production banking system. Results depend on the dataset, split, threshold, and model settings. The synthetic demo is not evidence of real-world performance. Deployment would require additional validation, monitoring, security controls, and privacy safeguards.

## Author

**Abeeha Zeeshan**  
Student Researcher | AI & Cybersecurity

## Live Demo

Add the Streamlit deployment URL here after deployment.

## License and Data Attribution

Check the dataset's current license and terms before redistributing data or publishing derived artifacts.
