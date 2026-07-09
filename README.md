# Sistem Deteksi Kemiripan Semantik KTI PMG BMKG

> Tesis S2 Teknik Informatika — Dhedy Listyawan (NIM 241012000180)
> Universitas Pamulang — 2026

Aplikasi web deteksi kemiripan semantik Karya Tulis Ilmiah (KTI)
Jabatan Fungsional PMG BMKG berbasis **Siamese LSTM** dan
**Siamese IndoBERT**.

| Model | F1-Score | Accuracy | Ukuran | GPU |
|---|---|---|---|---|
| Siamese LSTM ⭐ | 69,26% | 93,68% | 16,77 MB | Tidak perlu |
| Siamese IndoBERT | 63,68% | 92,79% | 498 MB | Disarankan |

---

## Struktur Folder

```
streamlit_app/
├── app.py                    ← Aplikasi utama
├── requirements.txt          ← Dependensi Python
├── README.md                 ← Panduan ini
├── .streamlit/
│   └── config.toml           ← Konfigurasi Streamlit
└── models/                   ← Letakkan file model di sini
    ├── best_model_v2.keras   ← Model Siamese LSTM (dari Google Drive)
    ├── train_pairs.csv       ← Data latih untuk tokenizer LSTM
    ├── best_model.pt         ← Model Siamese IndoBERT
    └── tokenizer/            ← Folder tokenizer IndoBERT
        ├── vocab.txt
        ├── tokenizer_config.json
        └── config.json
```

---

## Cara Menjalankan Lokal

### 1. Install dependensi

```bash
# Buat virtual environment (opsional tapi disarankan)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install packages
pip install -r requirements.txt
```

### 2. Siapkan file model

Salin file model dari Google Drive ke folder `models/`:

```bash
# Dari Google Colab atau Google Drive, download:
# - best_model_v2.keras  → models/
# - train_pairs.csv      → models/
# - best_model.pt        → models/
# - tokenizer/           → models/tokenizer/
```

### 3. Jalankan aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di: `http://localhost:8501`

---

## Deploy ke Streamlit Community Cloud (Gratis)

### Langkah 1 — Push ke GitHub

```bash
git init
git add .
git commit -m "Sistem Deteksi Kemiripan KTI PMG BMKG"
git remote add origin https://github.com/USERNAME/kti-pmg-sts.git
git push -u origin main
```

### Langkah 2 — Upload model ke GitHub LFS atau Google Drive

Karena model besar (best_model.pt = 498 MB), gunakan salah satu:

**Opsi A — GitHub LFS (untuk model LSTM saja, 16,77 MB):**
```bash
git lfs install
git lfs track "*.keras"
git lfs track "*.pt"
git add .gitattributes
git add models/
git commit -m "Add model files"
git push
```

**Opsi B — Download saat startup (untuk semua model):**
Tambahkan file `setup.sh` yang men-download model dari Google Drive
saat aplikasi pertama kali dijalankan.

### Langkah 3 — Deploy di share.streamlit.io

1. Buka https://share.streamlit.io
2. Login dengan akun GitHub
3. Klik "New app"
4. Pilih repository dan branch
5. Set "Main file path" → `app.py`
6. Klik "Deploy"

URL publik akan tersedia dalam 2-5 menit:
`https://USERNAME-kti-pmg-sts.streamlit.app`

---

## Panduan Pengguna Tim Penilai PMG BMKG

### Mode Demo 1-vs-1
1. Pilih model di sidebar (rekomendasi: Siamese LSTM)
2. Upload Dokumen KTI A (PDF)
3. Upload Dokumen KTI B (PDF)
4. Klik tombol "🔍 Analisis Kemiripan"
5. Baca hasil skor dan label MIRIP/TIDAK MIRIP

### Mode Evaluasi Batch
1. Download template CSV dari aplikasi
2. Isi CSV dengan pasangan KTI yang ingin dievaluasi
3. Masukkan semua PDF ke dalam satu file ZIP
4. Upload CSV dan ZIP ke aplikasi
5. Klik "▶️ Jalankan Evaluasi Batch"
6. Download hasil lengkap dalam format CSV

---

## Interpretasi Skor

| Skor | Interpretasi |
|---|---|
| 0,80 – 1,00 | **Sangat Mirip** — Perlu investigasi mendalam |
| 0,59 – 0,80 | **Mirip** — Perlu review manual Tim Penilai |
| 0,30 – 0,59 | Cukup Mirip — Di bawah threshold, kemungkinan kecil |
| 0,00 – 0,30 | **Tidak Mirip** — Konten semantik berbeda |

*Threshold 0,59 untuk Siamese LSTM, 0,75 untuk Siamese IndoBERT*

---

## Referensi

- Mueller & Thyagarajan (2016). Siamese Recurrent Architectures.
  AAAI 2016. DOI: 10.1609/aaai.v30i1.10350
- Wilie et al. (2020). IndoNLU. AACL-IJCNLP 2020.
  DOI: 10.18653/v1/2020.aacl-main.85
- Cer et al. (2017). SemEval-2017 Task 1.
  DOI: 10.18653/v1/S17-2001
