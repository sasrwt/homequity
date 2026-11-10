
# -*- coding: utf-8 -*-
"""
BUS 458: Analytics - From Data to Decisions
Streamlit Interactive Deployment Template: HMEQ Loan Evaluation
"""

import streamlit as st
import pickle
import pandas as pd
import sklearn 

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & MODEL LOADING
# ---------------------------------------------------------
st.set_page_config(page_title="HMEQ Loan Decision", page_icon="🏦", layout="wide")

# Updated way to load for performance (Essential for GitHub/Streamlit Cloud deployment)
@st.cache_resource
def load_model():
    with open("hmeq_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

st.title("Home Equity Loan Scoring Dashboard")
st.write("BUS 458 Interactive Decision Tool")
st.divider()

# ---------------------------------------------------------
# 2. INPUT FORM (3-COLUMN LAYOUT)
# ---------------------------------------------------------
st.subheader("Applicant Financial Profile")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### Loan & Property")
    loan = st.number_input("Loan Amount Requested ($)", min_value=1000, max_value=500000, value=20000, step=1000)
    mortdue = st.number_input("Existing Mortgage Due ($)", min_value=0.0, max_value=1000000.0, value=50000.0, step=1000.0)
    value = st.number_input("Property Assessed Value ($)", min_value=0.0, max_value=1000000.0, value=85000.0, step=1000.0)
    reason = st.selectbox("Reason for Loan", ["DebtCon", "HomeImp"])

with col2:
    st.markdown("##### Employment & Income")
    job = st.selectbox("Occupational Category", ["Office", "Mgr", "ProfExe", "Sales", "Self", "Other"])
    yoj = st.slider("Years at Current Job", min_value=0, max_value=45, value=5, step=1)
    debtinc = st.slider("Debt-to-Income Ratio (%)", min_value=0.0, max_value=205.0, value=35.0, step=0.1)

with col3:
    st.markdown("##### Credit History")
    clage = st.slider("Age of Oldest Trade Line (Months)", min_value=0.0, max_value=1200.0, value=180.0, step=1.0)
    clno = st.slider("Number of Open Credit Lines", min_value=0, max_value=75, value=15, step=1)
    ninq = st.number_input("Recent Credit Inquiries", min_value=0, max_value=20, value=0, step=1)
    delinq = st.number_input("Delinquent Credit Lines", min_value=0, max_value=20, value=0, step=1)
    derog = st.number_input("Major Derogatory Reports", min_value=0, max_value=20, value=0, step=1)

# ---------------------------------------------------------
# 3. DATA PREPROCESSING
# ---------------------------------------------------------
# Assemble the live user inputs into a 1-row DataFrame
input_data = pd.DataFrame({
    "LOAN": [loan],
    "MORTDUE": [mortdue],
    "VALUE": [value],
    "YOJ": [yoj],
    "DEROG": [derog],
    "DELINQ": [delinq],
    "CLAGE": [clage],
    "NINQ": [ninq],
    "CLNO": [clno],
    "DEBTINC": [debtinc],
    "REASON": [reason],
    "JOB": [job]
})

# One-hot encode categorical variables
input_data_encoded = pd.get_dummies(input_data, columns=["REASON", "JOB"])

# Align columns to match the exact schema the model saw during training
model_columns = model.feature_names_in_
for col in model_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# Reorder columns to match training order exactly
input_data_encoded = input_data_encoded[model_columns]

# ---------------------------------------------------------
# 4. PREDICTION & BUSINESS LOGIC
# ---------------------------------------------------------
st.divider()

if st.button("Evaluate Loan Application", type="primary", use_container_width=True):
    # Get both the 0/1 prediction AND the raw probability
    prediction = model.predict(input_data_encoded)[0]
    probabilities = model.predict_proba(input_data_encoded)[0]
    
    # probabilities[1] is the exact probability of a Bad Loan (Class 1)
    bad_loan_risk = probabilities[1] 
    
    # Display the calculated risk percentage 
    st.write(f"**Calculated Default Risk:** {bad_loan_risk*100:.1f}%")
    
    # Apply business threshold logic (overriding the rigid 50% default)
    if bad_loan_risk > 0.20:
        st.error("The prediction is: **Bad Loan** 🚫 (Declined due to risk > 20%)")
    else:
        st.success("The prediction is: **Good Loan** 💲 (Approved)")
