import streamlit as st
import pandas as pd
import joblib
import os
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# Define model path in a temporary directory
temp_dir = tempfile.gettempdir()
model_path = os.path.join(temp_dir, "default_risk_model.pkl")

# Load dataset and retrain model if the file is missing or corrupted
def train_and_save_model():
    df = pd.read_csv("model_data.csv")
    df = pd.get_dummies(df, columns=['gender', 'income_category'], drop_first=False)  # Keep all dummies
    X = df.drop(columns=['user_id', 'default_risk_score'])
    y = df['default_risk_score']
    column_order = X.columns.tolist()
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump((model, column_order), model_path)
    return model, column_order

# Try to load the model, retrain if corrupted
try:
    model, column_order = joblib.load(model_path)
except (FileNotFoundError, ValueError):  # Handle missing/corrupt file
    model, column_order = train_and_save_model()

# Load the offers data
home_deals = pd.read_csv("home_deals.csv")
phone_deals = pd.read_csv("Phone data.csv")

# Title and header
st.header('Unbanked Customer Affordability Modeling System')
st.write("Predict the default risk score based on user behavior and financial data.")

# Input widgets
age = st.slider('Age', 18, 70, 30)
gender = st.selectbox('Gender', ['Male', 'Female'])
banked = st.radio('Banked Status', ['Unbanked', 'Banked'])
avg_call_duration = st.number_input('Avg Call Duration Per Day (minutes)', min_value=0.0, max_value=60.0, step=0.1)
num_calls_per_day = st.number_input('Number of Calls Per Day', min_value=0, max_value=100, step=1)
num_sms_per_day = st.number_input('Number of SMS Per Day', min_value=0, max_value=100, step=1)
data_usage_mb = st.number_input('Daily Data Usage (MB)', min_value=0, max_value=5000, step=1)
monthly_topups = st.number_input('Monthly Top-ups Per Month', min_value=0, max_value=30, step=1)
avg_topup_amount = st.number_input('Average Top-up Amount Per Month', min_value=0, max_value=2000, step=1)
geographic_mobility = st.number_input('Geographic Mobility Score', min_value=0, max_value=10, step=1)
income_category = st.selectbox('Income Category', ['Low', 'Medium', 'High'])

# Convert categorical inputs to numeric
encoded_gender = [1, 0] if gender == 'Male' else [0, 1]  # One-hot encoding for gender
encoded_banked = 1 if banked == 'Banked' else 0
income_dummies = {'Low': [1, 0, 0], 'Medium': [0, 1, 0], 'High': [0, 0, 1]}
encoded_income = income_dummies[income_category]

# Prepare input data
input_data = pd.DataFrame([[age] + encoded_gender + [encoded_banked, avg_call_duration, num_calls_per_day, num_sms_per_day,
                            data_usage_mb, monthly_topups, avg_topup_amount, geographic_mobility] + encoded_income],
                          columns=column_order)

# Predict default risk score
if st.button('Predict Default Risk Score'):
    risk_score = model.predict(input_data)[0]
    st.success(f'Predicted Default Risk Score: {risk_score:.2f}')

    # Determine affordability based on spending habits and income category
    spending_score = avg_call_duration + num_calls_per_day + num_sms_per_day + data_usage_mb + (monthly_topups * avg_topup_amount)
    affordability_threshold = (1000 if income_category == 'Low' else 2000 if income_category == 'Medium' else 5000)
    
    st.write(f"Calculated Spending Score: {spending_score:.2f}")
    st.write(f"Affordability Threshold: {affordability_threshold}")
    
    if spending_score > affordability_threshold:
        st.error("High Spending - Limited approval for deals")
    else:
        st.success("Sufficient affordability - Eligible for more deals")
    
    # Define approval function
    def is_approved(deal_price):
        return deal_price <= (affordability_threshold - spending_score)
    
    # Filter and show approved home deals
    st.subheader("Recommended Home Internet Deals")
    for _, row in home_deals.iterrows():
        if is_approved(row['priceZAR']):
            st.image(row['image_url'], width=150)
            st.write(f"**{row['planName']}** - {row['provider']}")
            st.write(f"Speed: {row['speedMbps']} Mbps, Bandwidth: {row['bandwidthGB']} GB")
            st.write(f"Price: R{row['priceZAR']:.2f} (Approved)")
            st.write("---")
    
    # Filter and show approved mobile deals
    st.subheader("Recommended Mobile Deals")
    for _, row in phone_deals.iterrows():
        if is_approved(row['MonthyAmount']):
            st.image(row['Image_url'], width=150)
            st.write(f"**{row['Make']} {row['Model']} ({row['Variant']})** - {row['Colour']}")
            st.write(f"Contract Price: {row['ContractPriceDisplay']}, Monthly Payment: R{row['MonthyAmount']:.2f} (Approved)")
            st.write(f"Data: {row['DataDisplay']}, Minutes: {row['Minutes']} min")
            st.write("---")

# Display feature importance
if st.checkbox("Show Feature Importance"):
    feature_importances = pd.DataFrame({'Feature': column_order, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=feature_importances)
    plt.title('Feature Importance')
    st.pyplot(plt)
