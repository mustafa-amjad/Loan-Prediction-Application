import pandas as pd
import streamlit as st

from loan.model import approval_probability, load_data, train

st.set_page_config(page_title="Loan approval predictor", page_icon=":material/account_balance:", layout="wide")


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_data()


@st.cache_resource
def get_model():
    return train(get_data())


df = get_data()
model = get_model()

st.title("Loan approval predictor")
st.caption(
    "A Logistic Regression pipeline trained on 20,000 loan applications. "
    "Learning project; not for real lending decisions."
)

predict_tab, model_tab, data_tab = st.tabs(
    [":material/person_search: Predict", ":material/monitoring: Model", ":material/table: Data"]
)

with predict_tab:
    with st.form("applicant"):
        a, b, c = st.columns(3, gap="large")
        with a:
            st.subheader("Applicant")
            gender = st.segmented_control("Gender", ["M", "F"], default="M")
            age = st.slider("Age", 21, 60, 30)
            city = st.selectbox("City", sorted(df["city"].unique()))
            employment_type = st.selectbox("Employment", sorted(df["employment_type"].unique()))
        with b:
            st.subheader("Finances")
            monthly_income_pkr = st.number_input("Monthly income (PKR)", 15_000, 500_000, 120_000, 5_000)
            credit_score = st.slider("Credit score", 300, 900, 680)
            existing_loans = st.segmented_control("Existing loans", [0, 1, 2, 3], default=0)
            default_history = st.toggle("Has defaulted before")
            has_credit_card = st.toggle("Has a credit card")
        with c:
            st.subheader("Loan")
            bank = st.selectbox("Bank", sorted(df["bank"].unique()))
            loan_amount_pkr = st.number_input("Loan amount (PKR)", 50_000, 3_500_000, 800_000, 25_000)
            loan_tenure_months = st.segmented_control(
                "Tenure (months)", [6, 12, 24, 36, 48, 60], default=36
            )
        submitted = st.form_submit_button("Predict approval", type="primary", icon=":material/gavel:")

    if submitted:
        applicant = {
            "gender": gender or "M",
            "age": age,
            "city": city,
            "employment_type": employment_type,
            "bank": bank,
            "monthly_income_pkr": monthly_income_pkr,
            "credit_score": credit_score,
            "loan_amount_pkr": loan_amount_pkr,
            "loan_tenure_months": loan_tenure_months or 36,
            "existing_loans": existing_loans or 0,
            "default_history": int(default_history),
            "has_credit_card": int(has_credit_card),
        }
        prob = approval_probability(model, applicant)
        with st.container(border=True):
            st.metric("Approval probability", f"{prob:.0%}")
            if prob >= 0.5:
                st.badge("Likely approved", icon=":material/check:", color="green")
            else:
                st.badge("Likely rejected", icon=":material/close:", color="red")
            st.progress(prob)

with model_tab:
    st.caption("Scored on a 20% stratified holdout set.")
    with st.container(horizontal=True):
        for name, value in model.metrics.items():
            st.metric(name, f"{value:.1%}", border=True)
    st.subheader("Confusion matrix")
    cm = pd.DataFrame(
        model.confusion,
        index=["Actual: rejected", "Actual: approved"],
        columns=["Predicted: rejected", "Predicted: approved"],
    )
    st.dataframe(cm)

with data_tab:
    st.caption(f"{len(df):,} applications · {df['approved'].mean():.1%} approved")
    st.dataframe(df.drop(columns=["applicant_name"]), hide_index=True, height=420)

st.caption("Built by Mustafa Amjad · [Source code](https://github.com/mustafa-amjad/Loan-Prediction-Application)")
