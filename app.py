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
# Charger le pipeline
# ---------------------------------------
pipeline = joblib.load("chaabi_lld_ai_model.pkl")

# ---------------------------------------
# Titre
# ---------------------------------------
st.title("🚗 Chaabi LLD - Insurance Fraud Detection")

st.write("""
Cette application détecte automatiquement les dossiers
susceptibles d'être frauduleux grâce à un modèle **XGBoost**.
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

    df = pd.read_csv(uploaded_file)

    st.subheader("Aperçu du fichier")

    st.dataframe(df.head())

    if st.button("🚀 Lancer la prédiction"):

        try:

            # Prédictions
            predictions = pipeline.predict(df)

            probabilities = pipeline.predict_proba(df)[:, 1]

            # Résultats
            results = df.copy()

            results["Predicted_Fraud"] = predictions
            results["Fraud_Probability"] = probabilities

            # Statistiques
            total = len(results)
            fraud = int(predictions.sum())
            normal = total - fraud

            st.success("Prédiction terminée avec succès.")

            c1, c2, c3 = st.columns(3)

            c1.metric("Nombre de dossiers", total)
            c2.metric("Fraudes détectées", fraud)
            c3.metric("Dossiers normaux", normal)

            st.subheader("Résultats")

            st.dataframe(results)

            st.subheader("Fraudes détectées")

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
