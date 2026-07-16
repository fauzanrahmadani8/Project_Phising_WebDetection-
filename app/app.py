"""
Aplikasi Streamlit - Sistem Deteksi Website Phishing
Capstone Project UAS Pembelajaran Mesin
"""
import streamlit as st

st.set_page_config(
    page_title="Deteksi Website Phishing",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🛡️ Deteksi Website Phishing")
st.sidebar.markdown("Capstone Project — UAS Pembelajaran Mesin")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Gunakan menu **Pages** di sidebar untuk berpindah antar halaman:\n"
    "- 📊 Dashboard EDA\n"
    "- 🔍 Model Demo\n"
    "- 📈 Evaluasi Model\n"
    "- 🧠 Interpretasi Hasil\n"
    "- 📖 Dokumentasi"
)

st.title("🛡️ Sistem Deteksi Website Phishing")
st.markdown(
    """
Selamat datang di aplikasi **Sistem Deteksi Website Phishing** berbasis Machine Learning.

Aplikasi ini dibangun sebagai bagian dari Capstone Project mata kuliah **Pembelajaran Mesin**,
yang mendemonstrasikan pipeline Machine Learning end-to-end: mulai dari akuisisi data, eksplorasi
dan preprocessing, pemodelan, evaluasi, hingga deployment.

### Fitur Aplikasi
| Halaman | Deskripsi |
|---|---|
| 📊 **Dashboard EDA** | Visualisasi interaktif hasil eksplorasi data |
| 🔍 **Model Demo** | Input fitur URL secara manual untuk mendapatkan prediksi |
| 📈 **Evaluasi Model** | Metrik performa dan visualisasi evaluasi model |
| 🧠 **Interpretasi Hasil** | Feature importance dan insight bisnis |
| 📖 **Dokumentasi** | Penjelasan dataset, metodologi, dan cara penggunaan |

### Ringkasan Model
Model terbaik (**Random Forest, tuned**) mencapai performa berikut pada data uji:
"""
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", "96.2%")
col2.metric("Precision", "95.8%")
col3.metric("Recall", "96.6%")
col4.metric("AUC-ROC", "0.993")

st.info(
    "👈 Gunakan menu di sidebar untuk menjelajahi setiap halaman aplikasi.",
    icon="ℹ️",
)
