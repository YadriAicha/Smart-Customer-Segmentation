import streamlit as st
import pandas as pd
import joblib


# ============================================================
# Configuration de la page
# ============================================================

st.set_page_config(
    page_title="Segmentation des clients",
    page_icon="👥",
    layout="centered"
)


# ============================================================
# Chargement du modèle
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load("rfm_random_forest.joblib")
    return model


rf_model = load_model()


# ============================================================
# Titre
# ============================================================

st.title("👥 Segmentation des clients")
st.write(
    "Entrez les informations RFM d'un client "
    "pour prédire son segment."
)


# ============================================================
# Saisie des données RFM
# ============================================================

recency = st.number_input(
    "Recency",
    min_value=0.0,
    value=30.0,
    step=1.0
)

frequency = st.number_input(
    "Frequency",
    min_value=0.0,
    value=5.0,
    step=1.0
)

monetary = st.number_input(
    "Monetary",
    min_value=0.0,
    value=100.0,
    step=10.0
)


# ============================================================
# Prediction
# ============================================================

if st.button("Prédire le segment"):

    nouveau_client = pd.DataFrame({
        "recency": [recency],
        "frequency": [frequency],
        "monetary": [monetary]
    })

    prediction = rf_model.predict(nouveau_client)

    cluster = prediction[0]


    # ========================================================
    # Nom du segment
    # ========================================================

    segments = {
        0: "client à forte valeur",
        1: "clients occasionnels",
        2: "clients intermédiaires"
    }


    nom_segment = segments.get(
        cluster,
        "Segment inconnu"
    )


    # ========================================================
    # Résultats
    # ========================================================

    st.success(f"Cluster prédit : {cluster}")

    st.subheader("Nom du segment")
    st.write(nom_segment)

    st.subheader("Informations RFM saisies")

    rfm_result = pd.DataFrame({
        "Variable": [
            "Recency",
            "Frequency",
            "Monetary"
        ],
        "Valeur": [
            recency,
            frequency,
            monetary
        ]
    })

    st.dataframe(
        rfm_result,
        hide_index=True
    )