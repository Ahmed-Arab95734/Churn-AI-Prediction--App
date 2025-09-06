import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =========================
# Titles & Introduction
# =========================
st.title("📊 Customer Churn Prediction")
st.markdown(
    """
    This interactive web app predicts whether a customer is likely to **churn**  
    (leave the company) or **stay** using an Artificial Intelligence model.  

    ✅ **Churn** means when a customer stops using a company’s services.  
    ✅ **Tenure** means how long the customer has been with the company (in months).  
    """
)

st.markdown("---")
st.subheader("👤 About This Project")
st.write("Developed as part of the ITI Data Mining Course project by **Ahmed Arab**.")

# =========================
# Load Model
# =========================
# Load the trained machine learning model
model = joblib.load("RandomForestClassifier8.pkl")

# Define labels for prediction results
Churn_labels = {0: "No (Customer will stay)", 1: "Yes (Customer will churn)"}

# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("🔧 Input Customer Information")

# Input fields for the model
Age = st.sidebar.slider("Age (years)", 18, 100, 22)
Gender = st.sidebar.selectbox("Gender", ["Male", "Female"],index=1)
Tenure = st.sidebar.slider("Tenure (months)", 0, 72, 59)
Usage_Frequency = st.sidebar.slider("Usage Frequency (per week)", 0, 30, 4)
Support_Calls = st.sidebar.slider("Support Calls (per day)", 0, 10, 3)
Payment_Delay = st.sidebar.slider("Payment Delay (days)", 0, 60, 20)
Subscription_Type = st.sidebar.selectbox("Subscription Type", ["Basic", "Standard", "Premium"],index=2)
Contract_Length = st.sidebar.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"],index=1)
Total_Spend = st.sidebar.slider("Total Spend ($)", 0, 1000, 501)
Last_Interaction = st.sidebar.slider("Last Interaction (days ago)", 0, 30, 15)

# Prepare input for prediction
input_data = np.array([[
    Age,
    1 if Gender == "Male" else 0,
    Tenure,
    Usage_Frequency,
    Support_Calls,
    Payment_Delay,
    0 if Subscription_Type == "Basic" else 1 if Subscription_Type == "Standard" else 2,
    0 if Contract_Length == "Monthly" else 1 if Contract_Length == "Quarterly" else 2,
    Total_Spend,
    Last_Interaction
]])

# =========================
# Model Prediction
# =========================
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)[0]

# =========================
# Show Results
# =========================
st.markdown("---")
st.subheader("📌 Prediction Result")
st.success(f"**Predicted Churn:** {Churn_labels[prediction]}")

# =========================
# Probabilities + Chart Side by Side
# =========================
st.subheader("📊 Churn Probability")

col1, col2 = st.columns([2, 1])  # wider column for text, smaller for chart

with col1:
    st.write(f"🟢 Stay (No Churn): **{prediction_proba[0]:.2%}**")
    st.write(f"🔴 Leave (Churn): **{prediction_proba[1]:.2%}**")

with col2:
    fig, ax = plt.subplots(figsize=(2, 2))  # smaller chart
    labels = ["Stay (No)", "Leave (Yes)"]
    colors = ["#4CAF50", "#FF5252"]
    ax.pie(prediction_proba, labels=labels, autopct="%1.1f%%",
           startangle=90, colors=colors, textprops={"fontsize": 8})
    ax.axis("equal")

    st.pyplot(fig)

# =========================
# Background Image with Light Text and Dark Sidebar Widgets
# =========================
page_bg_img = """
<style>
/* Main app background */
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.pexels.com/photos/8386440/pexels-photo-8386440.jpeg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    color: #f1f1f1;  /* light text */
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: rgba(0, 0, 0, 0.8); /* dark sidebar */
    color: #f1f1f1;
}

/* Sidebar selectbox, slider, etc. */
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
    background-color: #333333;   /* dark background */
    color: #f1f1f1;              /* light text */
    border-radius: 8px;
    border: 1px solid #555;
}

/* Dropdown menu options */
section[data-testid="stSidebar"] .stSelectbox div[role="listbox"] {
    background-color: #222222;
    color: #f1f1f1;
}

/* Label text */
section[data-testid="stSidebar"] label {
    color: #f1f1f1 !important;
}


/* Success box background (keep green) */
div.stAlert.success {
    background-color: #2e7d32 !important;
    border: 1px solid #1b5e20;
    border-radius: 8px;
}

/* Make all text inside success white */
div.stAlert.success p, 
div.stAlert.success span, 
div.stAlert.success strong {
    color: white !important;
}

/* Make the checkmark icon white */
div.stAlert.success [data-testid="stMarkdownContainer"]::before {
    filter: brightness(0) invert(1);
}

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)









