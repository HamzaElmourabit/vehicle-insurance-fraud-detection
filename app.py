import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------
# Configuration de la page
# ---------------------------------------
st.set_page_config(
    page_title="Chaabi LLD Fraud Detection",
    page_icon="🚗",
    layout="wide"
)

# ---------------------------------------
# Charger le modèle et les colonnes
# ---------------------------------------
model = joblib.load("chaabi_lld_fraud_detection_xgb (1).pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ---------------------------------------
# Titre
# ---------------------------------------
st.title("🚗 Chaabi LLD - Insurance Fraud Detection")

st.write("""
Cette application détecte automatiquement les dossiers suspects
de fraude grâce à un modèle **XGBoost**.
""")

st.divider()

# ---------------------------------------
# Upload du fichier CSV
# ---------------------------------------
uploaded_file = st.file_uploader(
    "📂 Importer un fichier CSV",
    type=["csv"]
)

if uploaded_file is not None:

    # Lecture du fichier
    df = pd.read_csv(uploaded_file)

    st.subheader("Aperçu du fichier")

    st.dataframe(df.head())

    if st.button("🚀 Lancer la prédiction"):

        try:

            # -------------------------------
            # Encodage automatique
            # -------------------------------
            df_encoded = pd.get_dummies(df)

            # Ajouter les colonnes manquantes
            for col in feature_columns:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0

            # Supprimer les colonnes en trop
            df_encoded = df_encoded.reindex(
                columns=feature_columns,
                fill_value=0
            )

            # -------------------------------
            # Prédictions
            # -------------------------------
            predictions = model.predict(df_encoded)

            probabilities = model.predict_proba(df_encoded)[:, 1]

            # -------------------------------
            # Résultats
            # -------------------------------
            results = df.copy()

            results["Predicted_Fraud"] = predictions
            results["Fraud_Probability"] = probabilities.round(4)

            total = len(results)
            fraud = int(predictions.sum())
            normal = total - fraud

            st.success("✅ Prédiction terminée avec succès.")

            c1, c2, c3 = st.columns(3)

            c1.metric("Nombre de dossiers", total)
            c2.metric("Fraudes détectées", fraud)
            c3.metric("Dossiers normaux", normal)

            st.subheader("Résultats")

            st.dataframe(results)

            st.subheader("🚨 Fraudes détectées")

            st.dataframe(
                results[
                    results["Predicted_Fraud"] == 1
                ]
            )

            csv = results.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇ Télécharger les résultats",
                data=csv,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:

            st.error("Erreur pendant la prédiction.")

            st.exception(e)

else:

    st.info("Veuillez importer un fichier CSV.")
