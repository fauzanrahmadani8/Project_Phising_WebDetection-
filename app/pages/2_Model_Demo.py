import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

st.set_page_config(page_title="Model Demo", page_icon="🔍", layout="wide")
st.title("🔍 Model Demo - Prediksi Status Website")

BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "..", "..", "models", "best_random_forest.pkl")
FEATCOLS_PATH = os.path.join(BASE, "..", "..", "models", "feature_columns.pkl")
STATS_PATH = os.path.join(BASE, "..", "..", "data", "processed", "feature_stats.json")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATCOLS_PATH)
    with open(STATS_PATH) as f:
        stats = json.load(f)
    return model, feature_cols, stats


try:
    model, feature_cols, stats = load_artifacts()
except FileNotFoundError:
    st.error("Model belum ditemukan. Jalankan notebook 01_eda_modeling.ipynb terlebih dahulu untuk melatih & menyimpan model.")
    st.stop()

st.markdown(
    """
Halaman ini memungkinkan Anda memasukkan nilai beberapa fitur **paling berpengaruh** (berdasarkan
feature importance model) secara manual. Fitur lainnya yang tidak ditampilkan akan otomatis diisi
dengan **nilai median** dari data latih, agar prediksi tetap dapat dihasilkan tanpa perlu mengisi
seluruh 87 fitur satu per satu.
"""
)

# Fitur-fitur paling penting yang ditampilkan sebagai input interaktif
top_features_info = [
    ("google_index", "Apakah domain terindeks Google?", "binary"),
    ("page_rank", "Page Rank domain (0-10)", "int"),
    ("nb_hyperlinks", "Jumlah hyperlink pada halaman", "int"),
    ("web_traffic", "Peringkat traffic web (semakin kecil = semakin populer)", "int"),
    ("nb_www", "Jumlah kemunculan 'www' pada URL", "int"),
    ("domain_age", "Usia domain (hari)", "int"),
    ("length_url", "Panjang URL (karakter)", "int"),
    ("phish_hints", "Jumlah kata mencurigakan pada URL", "int"),
    ("ratio_extHyperlinks", "Rasio hyperlink eksternal", "float"),
    ("safe_anchor", "Rasio anchor 'aman' (%)", "float"),
    ("nb_dots", "Jumlah titik (.) pada URL", "int"),
    ("ip", "Apakah hostname berupa alamat IP?", "binary"),
]

st.subheader("Input Fitur Utama")
input_vals = {}
cols = st.columns(3)
for i, (fname, label, ftype) in enumerate(top_features_info):
    col = cols[i % 3]
    default = stats["median"].get(fname, 0)
    fmin = stats["min"].get(fname, 0)
    fmax = stats["max"].get(fname, 100)
    with col:
        if ftype == "binary":
            val = st.selectbox(label, options=[0, 1], index=int(default), key=fname)
        elif ftype == "int":
            val = st.number_input(label, min_value=float(fmin), max_value=float(fmax) if fmax > fmin else float(fmin) + 1,
                                   value=float(default), step=1.0, key=fname)
        else:
            val = st.number_input(label, min_value=float(fmin), max_value=float(fmax) if fmax > fmin else float(fmin) + 1,
                                   value=float(default), key=fname)
    input_vals[fname] = val

st.markdown("---")

if st.button("🔮 Prediksi Status Website", type="primary", use_container_width=True):
    row = {}
    for fc in feature_cols:
        row[fc] = input_vals.get(fc, stats["median"].get(fc, 0))
    X_input = pd.DataFrame([row])[feature_cols]

    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0]

    label = "🚨 PHISHING" if pred == 1 else "✅ LEGITIMATE"
    conf = prob[pred] * 100

    c1, c2 = st.columns([1, 1])
    with c1:
        if pred == 1:
            st.error(f"### Hasil Prediksi: {label}")
        else:
            st.success(f"### Hasil Prediksi: {label}")
        st.metric("Tingkat Keyakinan Model", f"{conf:.2f}%")
    with c2:
        st.markdown("**Probabilitas per Kelas:**")
        prob_df = pd.DataFrame({
            "Kelas": ["Legitimate", "Phishing"],
            "Probabilitas": [prob[0], prob[1]]
        })
        st.bar_chart(prob_df.set_index("Kelas"))

    st.caption(
        "Catatan: Fitur yang tidak ditampilkan pada form diisi dengan nilai median dari data latih. "
        "Untuk hasil prediksi yang lebih akurat pada URL nyata, seluruh 87 fitur idealnya diekstraksi langsung dari URL/HTML target."
    )
