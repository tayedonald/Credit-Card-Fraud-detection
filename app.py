# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load
import matplotlib.pyplot as plt
import hashlib

# Security: Simple login mechanism
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

# Dummy user database
user_db = {"admin": make_hashes("password123")}

# User login function
def login():
    st.subheader("Login")
    username = st.text_input("User Name")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        if username in user_db and check_hashes(password, user_db[username]):
            st.success(f"Welcome {username}")
            return True
        else:
            st.error("Invalid username or password")
            return False
    return False

# Set page configuration
st.set_page_config(page_title="Banking App - Credit Card Fraud Detection", layout="wide")

# Load or train the model
def load_or_train_model(df):
    model_path = "fraud_detection_model.joblib"

    if st.checkbox("Retrain Model"):
        # Preprocess data
        X = df.drop("Class", axis=1)
        y = df["Class"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Save model
        dump(model, model_path)

        # Evaluate model
        y_pred = model.predict(X_test)
        st.write("### Model Evaluation:")
        st.write("#### Classification Report")
        st.text(classification_report(y_test, y_pred))
        st.write("#### Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))
    else:
        try:
            model = load(model_path)
        except FileNotFoundError:
            st.error("No saved model found. Please check the `Retrain Model` checkbox to train a new model.")
            return None

    return model

# Predict fraudulent transactions
def predict_fraud(model, input_data):
    try:
        prediction = model.predict(input_data)
        return prediction
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Data analysis and visualization function
def data_analysis(df):
    fraud_cases = df[df["Class"] == 1]
    non_fraud_cases = df[df["Class"] == 0]

    st.write("### Fraud vs Non-Fraud Transactions")
    st.bar_chart(df["Class"].value_counts())

    st.write("### Time Distribution of Fraudulent Transactions")
    fig, ax = plt.subplots()
    ax.hist(fraud_cases["Time"], bins=50, alpha=0.7, label="Fraud")
    ax.hist(non_fraud_cases["Time"], bins=50, alpha=0.7, label="Non-Fraud")
    ax.legend()
    st.pyplot(fig)

# About section
def about():
    st.write("# About the Banking App")
    st.write("""
    ## Credit Card Fraud Detection Banking App
    This application helps identify credit card fraud by leveraging machine learning models to classify transactions as fraudulent or non-fraudulent.
    Features include:
    - **User Authentication:** Simple login mechanism for secure access.
    - **Model Training:** Retrain the RandomForest model using new datasets.
    - **Fraud Detection:** Predict the fraudulent nature of credit card transactions.
    - **Visualization:** Provides graphical representation of fraud vs. non-fraud transactions.

    By Donald Taye
    """)

# Data exploration section
def data_exploration():
    st.write("# Data Exploration for Credit Card Fraud Detection")
    st.write("Upload your local credit card transaction CSV dataset below (default dataset provided).")
    uploaded_file = st.file_uploader("Select Your Local CSV Dataset", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df

    if "df" in st.session_state:
        df = st.session_state["df"]
        st.write("### Dataset Overview")
        st.write(df.head())
        st.write("### Dataset Summary")
        st.write(df.describe())

        data_analysis(df)

# Fraud prediction page with all specified features
def predict_page():
    st.markdown("<h1 style='text-align: center; color: white; background-color: blue;'>Fraud Detection System</h1>", unsafe_allow_html=True)
    st.write("## Enter Card Details for Fraud Prediction")

    transaction_id = st.text_input("Transaction ID")
    amount = st.number_input("Amount", min_value=0.0, step=0.01)
    time = st.number_input("Time", min_value=0.0, step=1.0)
    features = {f"V{i}": st.number_input(f"V{i}", value=0.0) for i in range(1, 29)}

    # Include the 'Time' column and other specified features
    transaction = {
        "id": transaction_id,
        "Time": time,
        **features,
        "Amount": amount
    }

    if st.button("Predict Fraud"):
        if all([transaction_id, *features.values(), amount, time]):
            # Conversion to DataFrame to match model input format
            transaction_df = pd.DataFrame([transaction])

            # Ensure 'Class' column is not included in the prediction input
            if "model" in st.session_state:
                model = st.session_state["model"]
                prediction = predict_fraud(model, transaction_df)

                if prediction is not None and prediction[0] == 1:
                    st.warning("**Fraudulent Transaction Detected!** The card is temporarily limited.")
                else:
                    st.success("**Transaction is Non-Fraudulent.** The card is allowed to proceed.")
            else:
                st.warning("Model not loaded. Please explore data and train the model first.")
        else:
            st.error("Please fill in all transaction details.")

# Dashboard navigation
def dashboard():
    st.sidebar.title("App Gallery")
    section = st.sidebar.radio("Navigation", ["About", "Login", "Data Exploration", "Predict Fraud"])

    if section == "About":
        about()
    elif section == "Login":
        if login():
            st.sidebar.success("Successfully logged in!")
            st.write("# Welcome to the Credit Card Fraud Detection App!")
            st.write("""
            After logging in, you can:
            - **Explore Data:** Upload and visualize your dataset.
            - **Predict Fraud:** Enter transaction details or upload a dataset to predict fraud.
            - **Train Models:** Retrain the fraud detection model with new datasets.
            """)
            st.write("## Please use the navigation menu to proceed.")
        else:
            st.sidebar.error("Please log in first.")
    elif section == "Data Exploration":
        data_exploration()
    elif section == "Predict Fraud":
        if "df" in st.session_state:
            st.session_state["model"] = load_or_train_model(st.session_state["df"])
            predict_page()
        else:
            st.warning("Please explore data first by uploading a dataset in the 'Data Exploration' section.")

# Main application execution
if "df" not in st.session_state:
    st.session_state["df"] = None

dashboard()
