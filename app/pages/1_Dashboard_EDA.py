import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Dashboard EDA", page_icon="📊", layout="wide")
st.title("📊 Dashboard EDA - Eksplorasi Data")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "dataset_phishing.csv")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()

st.markdown(f"Dataset terdiri dari **{df.shape[0]:,} baris** dan **{df.shape[1]} kolom** (87 fitur + kolom `url` + kolom target `status`).")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sampel", f"{df.shape[0]:,}")
col2.metric("Jumlah Fitur", f"{df.shape[1]-2}")
col3.metric("Missing Values", int(df.isnull().sum().sum()))
col4.metric("Data Duplikat", int(df.duplicated().sum()))

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Distribusi Kelas", "Distribusi Fitur", "Korelasi", "Data Mentah"])

with tab1:
    c1, c2 = st.columns([1, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x="status", data=df, hue="status", palette=["#E63946", "#2E86AB"], legend=False, ax=ax)
        ax.set_title("Distribusi Kelas Target")
        st.pyplot(fig)
    with c2:
        counts = df["status"].value_counts()
        pct = df["status"].value_counts(normalize=True) * 100
        st.dataframe(pd.DataFrame({"Jumlah": counts, "Persentase (%)": pct.round(2)}))
        st.success("Dataset seimbang sempurna (50:50), tidak diperlukan teknik balancing tambahan.")

with tab2:
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "status"]
    feature = st.selectbox("Pilih fitur untuk divisualisasikan:", numeric_cols, index=numeric_cols.index("length_url") if "length_url" in numeric_cols else 0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(data=df, x=feature, hue="status", bins=40, kde=True, palette=["#E63946", "#2E86AB"], ax=ax)
    ax.set_title(f"Distribusi {feature} berdasarkan Status")
    st.pyplot(fig)

    st.dataframe(df.groupby("status")[feature].describe())

with tab3:
    key_feats = ["length_url", "nb_hyperlinks", "web_traffic", "domain_age", "google_index",
                 "page_rank", "nb_www", "phish_hints", "ratio_extHyperlinks", "safe_anchor"]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df[key_feats].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Matrix - Fitur Kunci")
    st.pyplot(fig)

with tab4:
    st.dataframe(df.head(50), use_container_width=True)

st.markdown("---")
st.markdown("### 🔎 5 Insight Utama dari EDA")
st.markdown(
    """
1. **Dataset seimbang sempurna** (50% legitimate : 50% phishing), tanpa missing value atau duplikat.
2. **URL phishing rata-rata jauh lebih panjang** (~75 karakter) dibanding URL legitimate (~47 karakter).
3. Fitur `ratio_extHyperlinks` dan `safe_anchor` berkorelasi negatif — halaman phishing cenderung banyak link eksternal namun sedikit anchor "aman".
4. Website yang **tidak terindeks Google** (`google_index=0`) memiliki proporsi phishing yang sangat dominan.
5. Domain phishing memiliki **usia domain (domain_age) dan web traffic** yang jauh lebih rendah dibanding domain legitimate.
"""
)
