# FraudShield AI 🛡️
### AI-Powered Credit Card Fraud Detection Dashboard

FraudShield AI is a machine-learning project designed to explore how artificial intelligence can help identify potentially fraudulent credit card transactions. Built with Python, Scikit-learn, and Streamlit, it provides an interactive dashboard for exploring transaction data, examining fraud patterns, evaluating a machine-learning model, and testing sample transactions.

## 🚀 Live Demo

**[Launch FraudShield AI — Streamlit App](https://fraudshield-ai1.streamlit.app/)**

Open the live application to explore the dashboard and its available features.

## ✨ Features

- **Interactive dashboard:** View transaction statistics and fraud-related insights.
- **Dataset exploration:** Examine transaction records and the distribution of legitimate and fraudulent transactions.
- **Machine-learning classification:** Use a Random Forest classifier to classify transactions as potentially legitimate or fraudulent.
- **Model evaluation:** Review performance metrics, including accuracy, precision, recall, F1-score, and a confusion matrix.
- **Data visualizations:** Explore interactive charts to understand transaction patterns.
- **Sample transaction testing:** Test the prediction workflow using sample transaction data.
- **Data validation:** Check uploaded data for required columns and handle basic data-cleaning tasks.

## 🧠 How It Works

1. **Load transaction data:** The application loads the credit card transaction dataset or uses demonstration data when appropriate.
2. **Prepare the data:** It validates the data and separates the input features from the target label.
3. **Train the model:** A Random Forest classifier learns patterns from the training data.
4. **Evaluate performance:** The model is evaluated on held-out test data using classification metrics.
5. **Explore predictions:** Users can examine transaction patterns and test the prediction workflow through the dashboard.

## 📊 Dataset

This project uses the publicly available **Credit Card Fraud Detection** dataset from Kaggle.

- **Dataset:** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Total transactions:** 284,807
- **Legitimate transactions:** 284,315
- **Fraudulent transactions:** 492
- **Features:** 30 columns, including `Time`, `Amount`, anonymized features `V1`–`V28`, and the target label `Class`.

The dataset is highly imbalanced, with fraudulent transactions representing approximately 0.173% of all records. This makes precision, recall, and F1-score important when evaluating the model, rather than relying on accuracy alone.

**Dataset note:** The original CSV is not included in this repository because of its large file size. Download it from Kaggle and place `creditcard.csv` in the project folder if you want to run the application with the full dataset.

## 🛠️ Technologies Used

- **Python** — application logic and data processing
- **Streamlit** — interactive web application
- **Pandas** — data manipulation
- **NumPy** — numerical operations
- **Scikit-learn** — machine learning and evaluation metrics
- **Plotly** — interactive visualizations
- **Git and GitHub** — version control and project hosting

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/ab9eha/FraudShield-AI.git
cd FraudShield-AI
```


### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Add the dataset

Download the Kaggle dataset and place the extracted `creditcard.csv` file in the project folder, alongside `app.py`.

If the dataset is unavailable, the application may use its built-in demonstration data, depending on the application's configured fallback behavior.

### 4. Start the application

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser.

## 📁 Project Structure

```text
FraudShield-AI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── creditcard.csv  # Download separately; not committed
```

## 🎯 Project Goals

The goals of FraudShield AI are to:

- Explore the application of machine learning to financial fraud detection.
- Understand the challenges of highly imbalanced datasets.
- Practice data processing, model training, and evaluation.
- Build an interactive dashboard that communicates machine-learning results.
- Demonstrate an end-to-end AI project, from data preparation to web deployment.

## ⚠️ Limitations and Responsible Use

FraudShield AI is an educational and portfolio project. Its predictions should not be treated as confirmed fraud findings or used to make real financial decisions.

Model performance depends on the dataset, training process, and selected features. Results from this project do not establish effectiveness against real-world fraud. The application is not a production banking security system and should not be used to approve or decline actual transactions.

## 👩‍💻 Author

**Abeeha Zeeshan**

Student developer interested in Artificial Intelligence, Cybersecurity, and machine-learning applications.

## 📌 Project Status

The application has been deployed on Streamlit and is available for demonstration.

**Live application:** [https://fraudshield-ai1.streamlit.app/](https://fraudshield-ai1.streamlit.app/)

*Built as an educational AI and financial cybersecurity project.*