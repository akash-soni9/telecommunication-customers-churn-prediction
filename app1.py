import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

# Load model
model = joblib.load("final_gb_classifier.pkl")

# Preprocessing
def preprocess_input(data):
    df = pd.DataFrame(data, index=[0])

    df['InternetService'] = df['InternetService'].map({
        'DSL': 0, 'Fiber optic': 1, 'No': 2
    })
    df['Contract'] = df['Contract'].map({
        'Month-to-Month': 0, 'One year': 1, 'Two year': 2
    })
    df['PaymentMethod'] = df['PaymentMethod'].map({
        'Electronic check': 0,
        'Mailed check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 3
    })

    return df



st.markdown(
    """
    <h1 style="text-align:center;">📉 Customer Churn Prediction</h1>
    <p style="text-align:center; color:gray;">
    Predict whether a customer is likely to churn using service & billing details
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

st.markdown("### 👤 Customer Information")
c1, c2, c3 = st.columns(3)

with c1:
    gender = st.radio("Gender", [0, 1], horizontal=True)
    senior_citizen = st.radio("Senior Citizen", [0, 1], horizontal=True)

with c2:
    partner = st.radio("Partner", [0, 1], horizontal=True)
    dependents = st.radio("Dependents", [0, 1], horizontal=True)

with c3:
    tenure_group = st.number_input("Tenure Group", min_value=0, max_value=6)
    paperless_billing = st.radio("Paperless Billing", [0, 1], horizontal=True)

st.markdown("---")


st.markdown("### 📡 Services")
c4, c5, c6 = st.columns(3)

with c4:
    phone_service = st.radio("Phone Service", [0, 1], horizontal=True)
    multiple_lines = st.radio("Multiple Lines", [0, 1], horizontal=True)
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])

with c5:
    online_security = st.radio("Online Security", [0, 1, 2], horizontal=True)
    online_backup = st.radio("Online Backup", [0, 1, 2], horizontal=True)
    device_protection = st.radio("Device Protection", [0, 1, 2], horizontal=True)

with c6:
    tech_support = st.radio("Tech Support", [0, 1, 2], horizontal=True)
    streaming_tv = st.radio("Streaming TV", [0, 1], horizontal=True)
    streaming_movies = st.radio("Streaming Movies", [0, 1], horizontal=True)

st.markdown("---")


st.markdown("### 💳 Billing Information")
c7, c8, c9 = st.columns(3)

with c7:
    contract = st.selectbox("Contract", ['Month-to-Month', 'One year', 'Two year'])
    payment_method = st.selectbox(
        "Payment Method",
        ['Electronic check', 'Mailed check',
         'Bank transfer (automatic)', 'Credit card (automatic)']
    )

with c8:
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0)

with c9:
    total_charges = st.number_input("Total Charges", min_value=0.0)

st.markdown("---")

col_btn = st.columns([2, 1, 2])[1]
with col_btn:
    predict = st.button("**Predict Churn**", use_container_width=True)

if predict:
    user_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'tenure_group': tenure_group
    }

    processed_data = preprocess_input(user_data)
    prediction = model.predict(processed_data)
    probability = model.predict_proba(processed_data)[0][1]

    st.markdown("### Prediction Result")

    if prediction[0] == 1:
        st.error(
            f"⚠️ **High Churn Risk**\n\n"
            f"**Churn Probability:** {probability:.2%}"
        )
    else:
        st.success(
            f"✅ **Low Churn Risk**\n\n"
            f"**Churn Probability:** {probability:.2%}"
        )
