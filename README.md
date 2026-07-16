# 🛡️ Sistem Deteksi Website Phishing

Capstone Project — Ujian Akhir Semester Mata Kuliah **Pembelajaran Mesin**, Fakultas Ilmu Komputer, Universitas Dian Nuswantoro (Semester Genap 2025/2026).

Proyek ini membangun sistem klasifikasi Machine Learning untuk mendeteksi website **phishing** vs **legitimate**, lengkap dengan pipeline end-to-end (EDA → Preprocessing → Modeling → Evaluation → Deployment) dan aplikasi web interaktif berbasis **Streamlit**.

## 📊 Ringkasan Hasil

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| **Random Forest (Tuned)** ⭐ | **0.9619** | **0.9583** | **0.9659** | **0.9621** | **0.9934** |
| Random Forest (Baseline) | 0.9593 | 0.9549 | 0.9641 | 0.9595 | 0.9935 |
| Decision Tree (Baseline) | 0.9339 | 0.9291 | 0.9396 | 0.9343 | 0.9339 |
| Decision Tree (Tuned) | 0.9318 | 0.9236 | 0.9414 | 0.9324 | 0.9525 |

Model terbaik: **Random Forest (Tuned)** — `max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=300`

## 📁 Struktur Repository

```
capstone-project-phishing-detection/
├── data/
│   ├── raw/                       # Data mentah (dataset_phishing.csv)
│   └── processed/                 # Hasil pipeline (results.json, model_comparison.csv, feature_stats.json)
├── notebooks/
│   └── 01_eda_modeling.ipynb      # EDA, preprocessing, modeling, evaluasi (sudah dieksekusi penuh)
├── src/
│   └── run_pipeline.py            # Script pipeline end-to-end (dapat dijalankan ulang)
├── models/
│   ├── best_random_forest.pkl     # Model terbaik
│   ├── best_decision_tree.pkl     # Model pembanding
│   ├── scaler.pkl                 # StandardScaler
│   └── feature_columns.pkl        # Urutan kolom fitur
├── app/
│   ├── app.py                     # Entry point Streamlit
│   └── pages/
│       ├── 1_Dashboard_EDA.py
│       ├── 2_Model_Demo.py
│       ├── 3_Evaluasi_Model.py
│       ├── 4_Interpretasi_Hasil.py
│       └── 5_Dokumentasi.py
├── reports/
│   └── Laporan_Teknis_Deteksi_Phishing.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

## 📦 Dataset

**Web Page Phishing Detection Dataset** (Hannousse & Yahiouche, 2021)
- 11.430 URL, 87 fitur (56 struktur URL, 24 konten HTML, 7 reputasi domain eksternal)
- Seimbang 50:50 (5.715 legitimate, 5.715 phishing)
- Sumber: https://data.mendeley.com/datasets/c2gw7fy2j4/3

## 🚀 Cara Menjalankan

### 1. Instalasi dependencies
```bash
pip install -r requirements.txt
```

### 2. Menjalankan ulang pipeline (opsional, model sudah tersedia di `models/`)
```bash
cd src
python run_pipeline.py
```

### 3. Menjalankan aplikasi Streamlit
```bash
cd app
streamlit run app.py
```
Aplikasi akan terbuka di `http://localhost:8501` dengan 5 halaman:
- 📊 **Dashboard EDA** — eksplorasi data interaktif
- 🔍 **Model Demo** — coba prediksi status website
- 📈 **Evaluasi Model** — metrik & visualisasi performa
- 🧠 **Interpretasi Hasil** — feature importance & insight bisnis
- 📖 **Dokumentasi** — penjelasan lengkap proyek

### 4. Membuka notebook
```bash
jupyter notebook notebooks/01_eda_modeling.ipynb
```

## 🔑 Insight Utama

1. Dataset seimbang sempurna, tanpa missing value maupun duplikat.
2. URL phishing rata-rata jauh lebih panjang (74,87 karakter) dibanding legitimate (47,38 karakter).
3. Fitur reputasi domain (`google_index`, `page_rank`, `web_traffic`, `domain_age`) adalah prediktor terkuat.
4. Random Forest secara konsisten mengungguli Decision Tree pada seluruh metrik evaluasi.
5. Analisis SHAP memvalidasi feature importance sekaligus memberi interpretasi arah pengaruh tiap fitur.

## 📚 Referensi

- Hannousse, A. & Yahiouche, S. (2021). *Towards benchmark datasets for machine learning based website phishing detection: An experimental study*. Engineering Applications of Artificial Intelligence, 104, 104347.
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR.
- Lundberg, S. M. & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.

## 👤 Kontributor

Capstone Project — Mata Kuliah Pembelajaran Mesin, A11.4501–45XX
