import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

st.set_page_config(page_title="Interpretasi Hasil", page_icon="🧠", layout="wide")
st.title("🧠 Interpretasi Hasil & Insight Bisnis")

BASE = os.path.dirname(__file__)
RF_MODEL_PATH = os.path.join(BASE, "..", "..", "models", "best_random_forest.pkl")
FEATCOLS_PATH = os.path.join(BASE, "..", "..", "models", "feature_columns.pkl")

try:
    model = joblib.load(RF_MODEL_PATH)
    feature_cols = joblib.load(FEATCOLS_PATH)
except FileNotFoundError:
    st.error("Model belum ditemukan. Jalankan notebook 01_eda_modeling.ipynb terlebih dahulu.")
    st.stop()

fi = pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)

st.subheader("Top 15 Feature Importance — Random Forest (Tuned)")
top_n = st.slider("Jumlah fitur ditampilkan:", 5, 30, 15)

fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
sns.barplot(data=fi.head(top_n), x="Importance", y="Feature", color="#2E86AB", ax=ax)
ax.set_title(f"Top {top_n} Feature Importance")
st.pyplot(fig)

st.markdown("---")
st.subheader("📌 Interpretasi Bisnis")

insights = [
    ("google_index", "Website yang tidak terindeks Google sangat berasosiasi dengan phishing — indeksasi mesin pencari adalah sinyal reputasi domain terkuat."),
    ("page_rank", "Domain dengan page rank rendah/tidak dikenal berisiko lebih tinggi menjadi target atau pelaku phishing."),
    ("nb_hyperlinks", "Jumlah hyperlink pada halaman mencerminkan kompleksitas struktur konten; pola tertentu berkorelasi dengan halaman phishing."),
    ("web_traffic", "Traffic web yang rendah mengindikasikan domain baru atau tidak sah — situs legitimate umumnya memiliki traffic organik lebih tinggi."),
    ("domain_age", "Domain yang baru dibuat jauh lebih rentan digunakan untuk phishing dibanding domain yang sudah lama beroperasi."),
]

for feat, desc in insights:
    if feat in fi["Feature"].values:
        rank = fi.reset_index(drop=True)
        idx = rank[rank.Feature == feat].index[0] + 1
        st.markdown(f"**#{idx} — `{feat}`**: {desc}")

st.markdown("---")
st.info(
    "💡 **Rekomendasi untuk tim keamanan:** Prioritaskan validasi reputasi domain (indeksasi, page rank, "
    "traffic, usia domain) sebagai lapisan pertahanan pertama sebelum menganalisis struktur URL secara detail, "
    "karena fitur reputasi domain memberikan kontribusi prediktif paling besar pada model."
)
