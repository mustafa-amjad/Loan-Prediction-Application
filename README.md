# Loan Approval Prediction

A Streamlit app that predicts whether a loan application will be approved, using a scikit-learn pipeline trained on applicant data (income, credit score, loan amount, tenure, employment and more).

> Built as a learning project, with Pakistani cities, banks and PKR amounts. Don't use it for real lending decisions.

## Features

- **End-to-end ML pipeline:** imputation, scaling and one-hot encoding in a single `Pipeline` / `ColumnTransformer`, so training and prediction always use the same steps
- **Logistic Regression** classifier with a holdout test set
- **Live metrics:** accuracy, precision, recall, F1 and the confusion matrix
- **Interactive prediction form** that returns an approve/reject decision with its probability
- Browse the full dataset in the app

## Dataset

`data/loan_dataset.csv` (20,000 applications) columns: gender, age, city, employment type, bank, monthly income (PKR), credit score, loan amount (PKR), loan tenure, existing loans, default history, has credit card. The target is `approved` (0/1).

## Project structure

```
Loan-Prediction-Application/
├── app.py              # Streamlit UI: predict, model metrics, data
├── loan/model.py       # training pipeline and prediction
├── data/loan_dataset.csv
├── .streamlit/config.toml
└── requirements.txt
```

## Tech stack

Python · pandas · NumPy · scikit-learn · Streamlit

## Run it locally

```bash
git clone https://github.com/mustafa-amjad/Loan-Prediction-Application.git
cd Loan-Prediction-Application
python -m venv venv
venv\Scripts\activate          # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

## Author

**Mustafa Amjad**, Data Scientist / AI Engineer · [LinkedIn](https://www.linkedin.com/in/mustafaamjad9)

MIT License
