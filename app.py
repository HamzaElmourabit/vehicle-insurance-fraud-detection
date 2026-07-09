
import streamlit as st
import pandas as pd
import joblib


# Charger le modèle
model = joblib.load("fraud_detection_xgb.pkl")


st.title("🚗 Chaabi LLD - Fraud Detection AI")

st.write(
    "Application de détection de fraude basée sur XGBoost"
)


st.sidebar.header("Informations du dossier")


# Exemple avec tes variables principales

Month = st.sidebar.number_input("Month", 0, 12, 1)
Make = st.sidebar.number_input("Make", 0, 10, 1)
AccidentArea = st.sidebar.number_input("Accident Area", 0, 5, 1)
Sex = st.sidebar.number_input("Sex", 0, 5, 1)
MaritalStatus = st.sidebar.number_input("Marital Status", 0, 5, 1)
BasePolicy = st.sidebar.number_input("Base Policy", 0, 5, 1)


if st.button("Predict Fraud"):

    data = pd.DataFrame({
        "Month":[Month],
        "Make":[Make],
        "AccidentArea":[AccidentArea],
        "Sex":[Sex],
        "MaritalStatus":[MaritalStatus],
        "BasePolicy":[BasePolicy]
    })


    prediction = model.predict(data)

    probability = model.predict_proba(data)[0][1]


    st.subheader("Résultat")


    if prediction[0] == 1:
        st.error(
            f"⚠️ Fraude détectée\nProbabilité : {probability:.2%}"
        )

    else:
        st.success(
            f"✅ Dossier normal\nProbabilité fraude : {probability:.2%}"
        )
