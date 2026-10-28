import joblib
import streamlit as st
import pandas as pd

st.title("How much is your car worth?")
st.write("Tell us about your car")

def hentData():
    return pd.read_csv("finn-annonser-data.csv")

model = joblib.load("bilmodell.joblib")
encoder = joblib.load("merke_encoder.joblib")
model_columns = joblib.load("model_columns.joblib")

df_data = hentData()

merker = sorted(df_data["merke"].unique())
merke = st.selectbox("Velg bilmerke", merker)

modeller_for_merke = sorted(
    set(df_data[df_data["merke"] == merke]["modell"])
)
modell = st.selectbox("Velg bilmodell", modeller_for_merke)

aarModell = st.number_input("Årsmodell", min_value=1940, max_value=2027, value=2026)
km = st.number_input("Kilometerstand", min_value=0, max_value=500000, value=0)
type = st.selectbox("Bil type", ["Bensin", "Diesel", "Elektrisk", "Hybrid"])

rekkevidde = 0
girkasse = "Automat"

if type in ["Elektrisk", "Hybrid"]:
    rekkevidde = st.number_input("Rekkevidde (km)", min_value=0, max_value=3000, value=0)

if type in ["Bensin", "Diesel"]:
    girkasse = st.selectbox("Girkasse", ["Automat", "Manuell"])

selger = st.selectbox("Selger", ["Privat", "Forhandler", "Merkeforhandler"])

if st.button("Beregn pris"):
    input_dict = {
        "aar": aarModell,
        "km": km,
        "rekkevidde": rekkevidde,
        "merke": merke,
        "modell": modell,
    }
    df_input = pd.DataFrame([input_dict])

    df_input[["merke_encoded", "modell_encoded"]] = encoder.transform(
        df_input[["merke", "modell"]])
    df_input = df_input.drop(columns=["merke", "modell"])

    for col in model_columns:
            if col not in df_input.columns:
                df_input[col] = 0

    type_col = f"type_{type}"
    if type_col in df_input.columns:
        df_input[type_col] = 1

    girkasse_col = f"girkasse_{girkasse}"
    if girkasse_col in df_input.columns:
        df_input[girkasse_col] = 1

    selger_col = f"selger_{selger}"
    if selger_col in df_input.columns:
        df_input[selger_col] = 1

    df_input = df_input[model_columns]

    estimert_pris = model.predict(df_input)[0]
    st.success(f"Estimert verdi: **{estimert_pris:,.0f} NOK**")
