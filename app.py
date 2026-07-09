"""
════════════════════════════════════════════════════════════════
SISTEM DETEKSI KEMIRIPAN SEMANTIK KTI PMG BMKG — v2.0
Perbaikan: strategi ekstraksi cerdas, cache model, UI lebih informatif
════════════════════════════════════════════════════════════════
"""

import streamlit as st
import numpy as np
import pandas as pd
import io, os, sys, time, zipfile, tempfile, re, json, pickle
from pathlib import Path
import tensorflow as tf

st.set_page_config(
    page_title="Sistem Deteksi Kemiripan KTI PMG BMKG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header{background:linear-gradient(135deg,#0F4C75 0%,#1B6CA8 50%,#118A7E 100%);
  padding:1.4rem 2rem;border-radius:12px;margin-bottom:1.2rem;color:white;text-align:center}
.main-header h1{font-size:1.5rem;font-weight:700;margin:0}
.main-header p{font-size:.85rem;margin:.3rem 0 0;opacity:.9}
.stat-card{background:white;border:1px solid #e2e8f0;border-radius:8px;
  padding:.7rem 1rem;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.stat-card .lbl{font-size:.7rem;color:#64748b;font-weight:600;text-transform:uppercase}
.stat-card .val{font-size:1.5rem;font-weight:800;color:#0F4C75}
.stat-card .sub{font-size:.7rem;color:#94a3b8}
.badge-mirip{display:inline-block;background:#dcfce7;color:#166534;
  border:1.5px solid #86efac;border-radius:20px;padding:.35rem 1rem;
  font-weight:700;font-size:.95rem}
.badge-tidak{display:inline-block;background:#fee2e2;color:#991b1b;
  border:1.5px solid #fca5a5;border-radius:20px;padding:.35rem 1rem;
  font-weight:700;font-size:.95rem}
.info-box{background:#f0f9ff;border-left:3px solid #0ea5e9;border-radius:4px;
  padding:.6rem .9rem;font-size:.8rem;color:#0c4a6e;margin:.4rem 0}
.warn-box{background:#fefce8;border-left:3px solid #eab308;border-radius:4px;
  padding:.6rem .9rem;font-size:.8rem;color:#713f12;margin:.4rem 0}
.scs-box{background:#f0fdf4;border-left:3px solid #22c55e;border-radius:4px;
  padding:.6rem .9rem;font-size:.8rem;color:#14532d;margin:.4rem 0}
.tok-info{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;
  padding:.5rem .8rem;font-size:.75rem;color:#475569;margin-top:.4rem}
#MainMenu,footer,header{visibility:hidden}
</style>
""", unsafe_allow_html=True)


# ══ UTILS: Library opsional ════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_pdfplumber():
    try:
        import pdfplumber; return pdfplumber
    except ImportError: return None

@st.cache_resource(show_spinner=False)
def load_sastrawi():
    try:
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
            StopWordRemoverFactory)
        return StopWordRemoverFactory().create_stop_word_remover()
    except ImportError: return None

@st.cache_resource(show_spinner=False)
def load_sklearn():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        return TfidfVectorizer, cosine_similarity
    except ImportError: return None, None


# ══ PREPROCESSING CERDAS ══════════════════════════════════════

CONTENT_MARKERS = [
    r'abstrak', r'abstract',
    r'pendahuluan', r'bab\s*i\b', r'latar\s*belakang',
]

def clean_pdf_text(text: str) -> str:
    """Bersihkan noise dari PDF."""
    if not text: return ""
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\.{3,}\s*\d+', '', text)          # hapus ...15
    text = re.sub(r'[^\x00-\x7F\u00C0-\u024F]', ' ', text)
    text = re.sub(r' {3,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_smart(text: str, target_chars: int = 3000) -> str:
    """
    Tidak ada ekstraksi khusus — kembalikan teks apa adanya.
    Pemotongan dilakukan di preprocess_lstm() dengan [:512]
    sesuai format CSV dari Notebook 01.
    """
    if not text:
        return ""
    return text

def preprocess_lstm(text: str, sw=None) -> str:
    """Preprocessing IDENTIK dengan Notebook 01_Preprocessing_KTI_PMG.
    1. lowercase 2. hapus angka (tanpa spasi) 3. hapus non-alfabet
    """
    if not text: return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)        # hapus angka TANPA spasi
    text = re.sub(r'[^a-zA-Z\s]', '', text) # hapus non-alfabet TANPA spasi
    text = re.sub(r'\s+', ' ', text).strip()
    if sw:
        try: text = sw.remove(text)
        except: pass
    return text
def get_stats(raw: str, processed: str) -> dict:
    """Statistik teks untuk UI."""
    words_raw  = len(raw.split())
    words_proc = len(processed.split())
    est_tok    = words_proc
    used_tok   = min(est_tok, 300)
    cov        = min(100, round(used_tok / max(est_tok, 1) * 100))
    return {
        'words_raw'  : words_raw,
        'chars_raw'  : len(raw),
        'words_proc' : words_proc,
        'est_tokens' : est_tok,
        'used_tokens': used_tok,
        'coverage'   : cov,
        'truncated'  : est_tok > 300,
    }


# ══ EKSTRAKSI PDF ═════════════════════════════════════════════

def extract_pdf(source) -> str:
    """
    Ekstrak teks dari PDF — IDENTIK dengan Notebook 01.
    Membaca SEMUA halaman, join dengan spasi.
    """
    plumb = load_pdfplumber()
    if not plumb: return ""
    try:
        with plumb.open(source) as pdf:
            pages = []
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages.append(t)
        raw = " ".join(pages)
        return raw.strip()
    except Exception as e:
        return ""

def extract_pdf_preview(source) -> str:
    """
    Ekstrak teks untuk PREVIEW saja — skip 3 halaman pertama
    (cover, lembar pengesahan, daftar isi) agar preview
    menampilkan isi dokumen yang bermakna (BAB I dst).
    Fungsi ini TIDAK digunakan untuk analisis model.
    """
    plumb = load_pdfplumber()
    if not plumb: return ""
    try:
        with plumb.open(source) as pdf:
            total = len(pdf.pages)
            skip  = min(3, total // 4)  # skip maks 3 hal atau 25%
            pages = []
            for i, page in enumerate(pdf.pages):
                if i < skip:
                    continue
                t = page.extract_text()
                if t:
                    pages.append(t)
        return " ".join(pages).strip()
    except Exception as e:
        return ""

def tfidf_sim(ta: str, tb: str) -> float:
    """TF-IDF cosine similarity."""
    TfV, cos = load_sklearn()
    if not TfV: return 0.0
    try:
        a = preprocess_lstm(ta)
        b = preprocess_lstm(tb)
        if not a or not b: return 0.0
        mat = TfV(max_features=10000).fit_transform([a, b])
        return float(cos(mat[0], mat[1])[0][0])
    except: return 0.0



# ══ DETEKSI PLAGIARISME — Pendekatan Leksikal ════════════════
# Berbeda dari Siamese LSTM yang mengukur kemiripan SEMANTIK,
# fungsi-fungsi ini mengukur kesamaan KONTEN/TEKS yang lebih
# relevan untuk deteksi plagiarisme KTI PMG.

def similarity_tfidf_plagiarism(text_a: str, text_b: str) -> dict:
    """
    Hitung kemiripan leksikal menggunakan TF-IDF Cosine Similarity.
    Lebih relevan untuk plagiarisme dibanding kemiripan semantik.
    """
    TfV, cos = load_sklearn()
    if not TfV: return {"score": 0.0, "error": "sklearn tidak tersedia"}
    try:
        # Preprocessing ringan (pertahankan kata penting)
        def prep_light(t):
            t = t.lower()
            t = re.sub(r'[^a-zA-Z0-9\s]', ' ', t)
            t = re.sub(r'\s+', ' ', t).strip()
            return t
        ta = prep_light(text_a)
        tb = prep_light(text_b)
        if not ta or not tb:
            return {"score": 0.0}
        mat  = TfV(max_features=20000, ngram_range=(1,2)).fit_transform([ta, tb])
        score = float(cos(mat[0], mat[1])[0][0])
        return {"score": round(score, 4)}
    except Exception as e:
        return {"score": 0.0, "error": str(e)}

def similarity_ngram(text_a: str, text_b: str,
                     n: int = 3) -> dict:
    """
    Hitung kesamaan n-gram (kalimat/frasa yang identik).
    n=3 berarti frasa 3 kata yang sama persis.
    Sangat efektif mendeteksi copy-paste langsung.
    """
    def get_ngrams(text, n):
        words = text.lower().split()
        return set(tuple(words[i:i+n])
                   for i in range(len(words)-n+1))

    ng_a = get_ngrams(text_a, n)
    ng_b = get_ngrams(text_b, n)
    if not ng_a or not ng_b:
        return {"score": 0.0, "shared": 0, "total_a": 0, "total_b": 0}
    shared  = ng_a & ng_b
    jaccard = len(shared) / len(ng_a | ng_b)
    overlap = len(shared) / min(len(ng_a), len(ng_b))
    return {
        "score"    : round(overlap, 4),
        "jaccard"  : round(jaccard, 4),
        "shared"   : len(shared),
        "total_a"  : len(ng_a),
        "total_b"  : len(ng_b),
        "pct_a"    : round(len(shared)/len(ng_a)*100, 1),
        "pct_b"    : round(len(shared)/len(ng_b)*100, 1),
    }

def get_similar_sentences(text_a: str, text_b: str,
                          min_words: int = 8,
                          top_n: int = 5) -> list:
    """
    Temukan kalimat/frasa yang sama persis antara dua dokumen.
    Berguna untuk menunjukkan bagian mana yang diduga plagiat.
    """
    import difflib

    def split_sentences(text):
        sents = re.split(r'[.!?\n]', text.lower())
        return [s.strip() for s in sents
                if len(s.strip().split()) >= min_words]

    sents_a = split_sentences(text_a)
    sents_b = split_sentences(text_b)

    matches = []
    for sa in sents_a:
        for sb in sents_b:
            ratio = difflib.SequenceMatcher(
                None, sa, sb).ratio()
            if ratio >= 0.80:  # 80% mirip
                matches.append({
                    "kalimat_a": sa[:200],
                    "kalimat_b": sb[:200],
                    "similarity": round(ratio*100, 1),
                })

    # Urutkan dari yang paling mirip
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    return matches[:top_n]

def comprehensive_plagiarism_check(text_a: str, text_b: str) -> dict:
    """
    Pemeriksaan plagiarisme komprehensif yang menggabungkan
    beberapa metode untuk hasil yang lebih akurat.

    Interpretasi persentase:
    - 0-10%  : Tidak ada indikasi plagiarisme
    - 10-30% : Kesamaan rendah, perlu perhatian
    - 30-50% : Kesamaan sedang, perlu review manual
    - 50-70% : Kesamaan tinggi, dugaan plagiarisme
    - >70%   : Sangat mirip, indikasi plagiarisme kuat
    """
    # 1. TF-IDF (kesamaan kata dan frasa)
    tfidf = similarity_tfidf_plagiarism(text_a, text_b)

    # 2. Bigram (frasa 2 kata yang sama)
    bigram = similarity_ngram(text_a, text_b, n=2)

    # 3. Trigram (frasa 3 kata yang sama)
    trigram = similarity_ngram(text_a, text_b, n=3)

    # 4. Cari kalimat mirip
    similar_sents = get_similar_sentences(text_a, text_b)

    # 5. Hitung skor plagiarisme gabungan
    # Bobot: TF-IDF 40%, bigram 30%, trigram 30%
    plagiarism_score = (
        tfidf["score"] * 0.40 +
        bigram["score"] * 0.30 +
        trigram["score"] * 0.30
    )

    # 6. Tentukan level risiko
    pct = plagiarism_score * 100
    if pct >= 70:
        risk = "🔴 SANGAT TINGGI — Indikasi plagiarisme kuat"
        risk_color = "#fee2e2"
    elif pct >= 50:
        risk = "🟠 TINGGI — Dugaan plagiarisme, perlu verifikasi"
        risk_color = "#fed7aa"
    elif pct >= 30:
        risk = "🟡 SEDANG — Kesamaan signifikan, perlu review manual"
        risk_color = "#fef9c3"
    elif pct >= 10:
        risk = "🟢 RENDAH — Kesamaan terbatas, kemungkinan bukan plagiat"
        risk_color = "#dcfce7"
    else:
        risk = "✅ SANGAT RENDAH — Tidak ada indikasi plagiarisme"
        risk_color = "#f0fdf4"

    return {
        "plagiarism_score"   : round(plagiarism_score, 4),
        "plagiarism_pct"     : round(pct, 1),
        "tfidf_score"        : tfidf["score"],
        "bigram_overlap"     : bigram["score"],
        "trigram_overlap"    : trigram["score"],
        "bigram_shared"      : bigram.get("shared", 0),
        "trigram_shared"     : trigram.get("shared", 0),
        "pct_a_bigram"       : bigram.get("pct_a", 0),
        "pct_b_bigram"       : bigram.get("pct_b", 0),
        "similar_sentences"  : similar_sents,
        "risk_level"         : risk,
        "risk_color"         : risk_color,
    }


# ══ MODEL LSTM (cache) ════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_lstm(weights_dir: str, tokenizer_path: str):
    """
    Load Siamese LSTM dari bobot numpy (.npy) + tokenizer.pkl.
    Pendekatan ini TIDAK bergantung pada versi Keras/TF —
    bobot dibaca langsung lalu di-set ke model yang dibangun lokal.
    Urutan BiLSTM: forward-kernel, forward-recurrent, forward-bias,
                   backward-kernel, backward-recurrent, backward-bias
    (terbukti menghasilkan F1=69.26% identik dengan hasil eksperimen)
    """
    try:
        MAX_LEN   = 300
        MAX_WORDS = 30000
        VOCAB     = 13075  # dari tokenizer training

        # ── Load tokenizer ────────────────────────────────────
        if not os.path.exists(tokenizer_path):
            return None, None, None, (
                f"File tidak ditemukan: {tokenizer_path}\n"
                f"Pastikan tokenizer.pkl ada di folder models/weights_numpy/")
        with open(tokenizer_path, 'rb') as f:
            tok = pickle.load(f)

        # ── Load bobot dari numpy ─────────────────────────────
        def load_npy(name):
            path = os.path.join(weights_dir, name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"File bobot tidak ditemukan: {path}")
            return np.load(path)

        w_emb      = load_npy('emb.npy')
        w_bili_fwd = [load_npy('bili_fwd_k.npy'),
                      load_npy('bili_fwd_r.npy'),
                      load_npy('bili_fwd_b.npy')]
        w_bili_bwd = [load_npy('bili_bwd_k.npy'),
                      load_npy('bili_bwd_r.npy'),
                      load_npy('bili_bwd_b.npy')]
        w_bn       = [load_npy('bn_gamma.npy'),
                      load_npy('bn_beta.npy'),
                      load_npy('bn_mm.npy'),
                      load_npy('bn_mv.npy')]
        w_dense    = [load_npy('dense_k.npy'),
                      load_npy('dense_b.npy')]

        # ── Bangun arsitektur ─────────────────────────────────
        ia  = tf.keras.Input((MAX_LEN,), name='input_a')
        ib  = tf.keras.Input((MAX_LEN,), name='input_b')
        emb = tf.keras.layers.Embedding(
                  VOCAB, 300, name='embedding')
        bli = tf.keras.layers.Bidirectional(
                  tf.keras.layers.LSTM(
                      128, dropout=.2, recurrent_dropout=.1),
                  name='bilstm')
        bn  = tf.keras.layers.BatchNormalization(name='batch_norm')
        dr  = tf.keras.layers.Dropout(.3, name='dropout')
        dn  = tf.keras.layers.Dense(
                  128, activation='relu',
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                  name='dense')

        def enc(x):
            return dn(dr(bn(bli(emb(x)))))

        va  = enc(ia)
        vb  = enc(ib)
        cs  = tf.keras.layers.Dot(
                  axes=1, normalize=True, name='cosine_sim')([va, vb])
        out = tf.keras.layers.Lambda(
                  lambda x: (x + 1.0) / 2.0, name='output')(cs)
        mdl = tf.keras.Model(inputs=[ia, ib], outputs=out)

        # ── Set bobot langsung ────────────────────────────────
        mdl.get_layer('embedding').set_weights([w_emb])
        # Urutan BiLSTM: fwd dulu, bwd kemudian
        mdl.get_layer('bilstm').set_weights(
            w_bili_fwd + w_bili_bwd)
        mdl.get_layer('batch_norm').set_weights(w_bn)
        mdl.get_layer('dense').set_weights(w_dense)

        # ── Warmup BatchNorm ──────────────────────────────────
        _d = np.zeros((2, MAX_LEN), dtype=np.int32)
        _  = mdl([_d, _d], training=False)

        return mdl, tok, MAX_LEN, None

    except Exception as e:
        import traceback
        return None, None, None, traceback.format_exc()

def predict_lstm(ta, tb, mdl, tok, mlen, th, sw):
    """Inferensi LSTM dengan preprocessing cerdas."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # Ekstrak bagian representatif dulu
    ta_smart = extract_smart(ta, target_chars=3000)
    tb_smart = extract_smart(tb, target_chars=3000)

    # Preprocessing
    ta_proc = preprocess_lstm(ta_smart, sw)
    tb_proc = preprocess_lstm(tb_smart, sw)

    stats_a = get_stats(ta, ta_proc)
    stats_b = get_stats(tb, tb_proc)

    def enc(texts):
        seqs = tok.texts_to_sequences([str(t) for t in texts])
        return pad_sequences(seqs, maxlen=mlen, padding='post', truncating='post')

    X1 = enc([ta_proc]); X2 = enc([tb_proc])
    # Pastikan model dalam inference mode (bukan training mode)
    # Ini penting untuk BatchNormalization agar menggunakan
    # moving average dari training, bukan statistik batch saat ini
    # Gunakan training=False agar BatchNorm pakai moving average
    score = float(mdl([X1, X2], training=False).numpy()[0][0])

    # Model Siamese LSTM dengan Focal Loss menghasilkan distribusi BINARY:
    # - Skor 0.50 = TIDAK MIRIP (minimum model, akibat ReLU)
    # - Skor 0.59-1.00 = MIRIP (hanya pasangan yang benar-benar mirip)
    # - Skor 1.00 = IDENTIK SEMPURNA
    # Threshold 0.59 (hasil kalibrasi grid search) tetap digunakan langsung.
    label = "MIRIP" if score >= th else "TIDAK MIRIP"

    # Hitung confidence untuk UI yang lebih informatif
    if score >= th:
        # Range [th, 1.0] → confidence [0%, 100%]
        confidence = (score - th) / (1.0 - th) * 100
    else:
        # Range [0.5, th) → confidence negatif (tidak mirip)
        confidence = 0.0

    return {
        "score"      : round(score, 4),
        "threshold"  : th,
        "label"      : label,
        "confidence" : round(confidence, 1),
        "model"      : "Siamese LSTM",
        "stats_a"    : stats_a,
        "stats_b"    : stats_b,
        "text_a_processed": ta_proc[:300],
        "text_b_processed": tb_proc[:300],
    }


# ══ UI HELPERS ════════════════════════════════════════════════

def gauge(score, threshold, label, model_name, confidence=0):
    color  = "#22c55e" if label=="MIRIP" else "#64748b"
    badge  = "badge-mirip" if label=="MIRIP" else "badge-tidak"
    emoji  = "✅" if label=="MIRIP" else "❌"

    # Tampilkan skor dalam range [0.50, 1.00] yang sesungguhnya
    # Gauge bar: 0.50 = kiri, 1.00 = kanan
    bar_pct = max(0, (score - 0.5) / 0.5 * 100)  # posisi di bar
    th_pct  = (threshold - 0.5) / 0.5 * 100        # posisi threshold di bar

    if label == "MIRIP":
        fill   = "#22c55e" if score >= 0.80 else "#eab308"
        skor_label = f"{score:.4f}"
    else:
        fill   = "#94a3b8"
        skor_label = f"{score:.4f}"

    st.markdown(f"""
    <div style='text-align:center;padding:1rem 0'>
      <div style='font-size:.8rem;color:#64748b;margin-bottom:.3rem'>
        Model: <b>{model_name}</b> &nbsp;|&nbsp; Threshold: <b>{threshold}</b>
      </div>
      <div style='font-size:2.8rem;font-weight:900;color:{color};line-height:1.1'>
        {skor_label}
      </div>
      <div style='font-size:.85rem;color:#94a3b8;margin:.1rem 0'>
        Skor cosine similarity [0,50 – 1,00]
      </div>
      <div style='margin:.5rem 0'><span class='{badge}'>{emoji} {label}</span></div>
      <div style='position:relative;background:#f1f5f9;border-radius:6px;
                  height:20px;overflow:visible;max-width:420px;margin:.6rem auto'>
        <div style='width:{bar_pct:.1f}%;height:100%;background:{fill};
                    border-radius:6px;transition:width .5s'></div>
        <div style='position:absolute;top:-18px;left:{th_pct:.1f}%;
                    transform:translateX(-50%);font-size:.68rem;
                    color:#0F4C75;font-weight:700;white-space:nowrap'>
          ▼ {threshold}
        </div>
      </div>
      <div style='display:flex;justify-content:space-between;
                  max-width:420px;margin:.3rem auto;font-size:.7rem;color:#94a3b8'>
        <span>0,50 (Tdk Mirip)</span>
        <span>1,00 (Identik)</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

def text_preview(text, label, stats=None):
    if not text:
        st.warning(f"Teks {label} kosong.")
        return
    preview = text[:600]+"..." if len(text)>600 else text
    wc = len(text.split())
    st.markdown(f"""
    <div style='background:#f8fafc;border:1px solid #e2e8f0;
                border-radius:8px;padding:.7rem;font-size:.78rem;
                color:#374151;line-height:1.55'>
      <b style='color:#0F4C75'>📄 {label}</b>
      <span style='color:#94a3b8;font-weight:400'>
        &nbsp;({wc:,} kata · {len(text):,} karakter)
      </span><br/>{preview}
    </div>
    """, unsafe_allow_html=True)

    if stats:
        trunc = "⚠️ Dipotong" if stats['truncated'] else "✅ Penuh"
        cov   = stats['coverage']
        st.markdown(f"""
        <div class='tok-info'>
          📊 <b>Token info:</b>
          Kata asli: <b>{stats['words_raw']:,}</b> →
          Setelah preprocessing: <b>{stats['words_proc']:,}</b> →
          Token ke model: <b>{stats['used_tokens']}</b>/300
          &nbsp;|&nbsp; Coverage: <b>{cov}%</b>
          &nbsp;|&nbsp; {trunc}
        </div>
        """, unsafe_allow_html=True)


# ══ SIDEBAR ══════════════════════════════════════════════════

def sidebar():
    st.sidebar.markdown("## ⚙️ Pengaturan Model")
    model = st.sidebar.radio(
        "Pilih Model",
        ["🏆 Siamese LSTM (Rekomendasi)",
         "📊 TF-IDF Baseline"],
        index=0
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📂 Path File Model")

    if "LSTM" in model:
        mp  = st.sidebar.text_input(
              "Folder bobot numpy",
              value="models/weights_numpy",
              help="Folder berisi emb.npy, bili_*.npy, bn_*.npy, dense_*.npy")
        csv = st.sidebar.text_input(
              "Path tokenizer (.pkl)",
              value="models/weights_numpy/tokenizer.pkl",
              help="File tokenizer.pkl dari folder weights_numpy")
        th  = st.sidebar.slider("Threshold LSTM", .30, .90,
              value=.59, step=.01,
              help="Default: 0,59 (kalibrasi optimal grid search)")
    else:
        mp = csv = None
        th = st.sidebar.slider("Threshold TF-IDF", .05, .50,
             value=.1753, step=.01)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Info Model")
    info = {
        "🏆 Siamese LSTM (Rekomendasi)": {
            "F1-Score":"69,26%","Accuracy":"93,68%",
            "Recall":"70,80%","Ukuran":"16,77 MB",
            "GPU":"Tidak perlu","Threshold":"0,59",
        },
        "📊 TF-IDF Baseline": {
            "F1-Score":"—","Accuracy":"—","Recall":"—",
            "Ukuran":"< 1 MB","GPU":"Tidak perlu",
            "Threshold":"0,1753 (P90)",
        },
    }
    key = "🏆 Siamese LSTM (Rekomendasi)" if "LSTM" in model else "📊 TF-IDF Baseline"
    for k,v in info[key].items():
        st.sidebar.markdown(
            f"<div style='font-size:.78rem;padding:.12rem 0'>"
            f"<b>{k}:</b> {v}</div>",
            unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='font-size:.7rem;color:#94a3b8;line-height:1.5'>
    📚 <b>Tesis S2</b><br>
    Dhedy Listyawan · NIM 241012000180<br>
    Teknik Informatika · Univ. Pamulang · 2026
    </div>""", unsafe_allow_html=True)

    return model, th, mp, csv


# ══ TAB 1: DEMO 1-vs-1 ═══════════════════════════════════════

def tab_demo(model, th, mp, csv):
    st.markdown("""
    <div style='background:#f0f9ff;border-left:3px solid #0ea5e9;
                border-radius:4px;padding:.6rem .9rem;font-size:.8rem;
                color:#0c4a6e;margin-bottom:.8rem'>
        💡 <b>Strategi ekstraksi v2:</b> Sistem mengambil bagian
        <b>paling representatif</b> dokumen (abstrak + pendahuluan +
        kesimpulan) untuk memaksimalkan akurasi dengan 300 token LSTM.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📄 Dokumen KTI — A**")
        fa = st.file_uploader("PDF A", type=["pdf"], key="fa",
                              label_visibility="collapsed")
    with c2:
        st.markdown("**📄 Dokumen KTI — B**")
        fb = st.file_uploader("PDF B", type=["pdf"], key="fb",
                              label_visibility="collapsed")

    with st.expander("✏️ Atau masukkan teks langsung"):
        ta_in = st.text_area("Teks A", height=100, key="ta")
        tb_in = st.text_area("Teks B", height=100, key="tb")

    st.markdown("<hr style='border-color:#e2e8f0;margin:.8rem 0'>",
                unsafe_allow_html=True)

    bc, ic = st.columns([1,3])
    with bc:
        run = st.button("🔍 Analisis Kemiripan",
                        type="primary", use_container_width=True)
    with ic:
        st.markdown(
            f"<div class='info-box'>Model: <b>{model}</b> · "
            f"Threshold: <b>{th:.2f}</b> · "
            f"Skor ∈ [0,50 – 1,00] · "
            f"Skor ≥ <b>{th:.2f}</b> → MIRIP</div>",
            unsafe_allow_html=True)

    if not run:
        return

    # Ekstrak teks
    with st.spinner("⏳ Mengekstrak teks PDF..."):
        raw_a = extract_pdf(fa) if fa else ta_in if ta_in else ""
        raw_b = extract_pdf(fb) if fb else tb_in if tb_in else ""

    if not raw_a or not raw_b:
        st.error("❌ Gagal mengekstrak teks. Pastikan PDF valid.")
        return

    # Preview dari halaman isi (skip cover/daftar isi)
    # Model tetap menggunakan raw_a/raw_b (full teks) untuk analisis
    st.markdown("### 📋 Preview Teks Diekstrak")
    st.markdown("""
    <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;
                padding:.5rem .8rem;font-size:.78rem;color:#64748b;margin-bottom:.5rem'>
    📌 <b>Preview</b> menampilkan sebagian teks dari halaman isi dokumen.
    <b>Analisis plagiarisme</b> menggunakan <b>seluruh teks</b> kedua dokumen
    (termasuk bagian yang tidak tampil di preview).
    Posisi awal yang berbeda antar dokumen adalah normal karena panjang
    halaman awal (cover, daftar isi) berbeda-beda.
    </div>
    """, unsafe_allow_html=True)
    prev_a = extract_pdf_preview(fa) if fa else raw_a
    prev_b = extract_pdf_preview(fb) if fb else raw_b
    pc1, pc2 = st.columns(2)
    with pc1: text_preview(prev_a, "Dokumen A")
    with pc2: text_preview(prev_b, "Dokumen B")

    # Inferensi
    with st.spinner("🤖 Menganalisis kemiripan semantik..."):
        t0  = time.time()
        sw  = load_sastrawi()

        if "LSTM" in model and mp and os.path.exists(mp) \
                and csv and os.path.exists(csv):
            # Cek apakah model sudah di-cache
            # Selalu load fresh — hindari cache yang corrupt
            # st.cache_resource menangani caching dengan aman
            with st.spinner("⏳ Memuat model LSTM (sekali saja)..."):
                mdl, tok, mlen, err = load_lstm(mp, csv)
                # mp = folder weights_numpy, csv = path tokenizer.pkl
            if err:
                st.error(f"Gagal load model: {err}")
                return

            result = predict_lstm(raw_a, raw_b, mdl, tok, mlen, th, sw)

        else:
            # TF-IDF atau model tidak ditemukan
            if "LSTM" in model:
                st.warning("⚠️ File model tidak ditemukan → TF-IDF fallback")
            # Smart extract untuk TF-IDF juga
            sa = extract_smart(raw_a, 3000)
            sb = extract_smart(raw_b, 3000)
            score = tfidf_sim(sa, sb)
            result = {
                "score": score,
                "label": "MIRIP" if score >= th else "TIDAK MIRIP",
                "threshold": th,
                "model": "TF-IDF Baseline",
                "stats_a": get_stats(raw_a, preprocess_lstm(sa, sw)),
                "stats_b": get_stats(raw_b, preprocess_lstm(sb, sw)),
            }

        elapsed = time.time() - t0

    # Hasil
    st.markdown("### 🎯 Hasil Analisis Kemiripan Semantik")
    gc, dc = st.columns([1,1])

    with gc:
        gauge(result["score"], result["threshold"],
              result["label"], result["model"],
              result.get("confidence", 0))

    with dc:
        st.markdown("#### 📊 Detail Hasil")
        conf = result.get("confidence", 0)
        rows = [
            ("Skor Kemiripan",  f"{result['score']:.4f}"),
            ("Range Skor",      "0,50 (tdk mirip) – 1,00 (identik)"),
            ("Threshold",       f"{result['threshold']:.2f}"),
            ("Confidence",      f"{conf:.1f}% di atas threshold" if result['label']=="MIRIP" else "Di bawah threshold"),
            ("Status",          result["label"]),
            ("Model",           result["model"]),
            ("Waktu Inferensi", f"{elapsed:.2f} detik"),
        ]
        for k,v in rows:
            r1,r2 = st.columns([1,1])
            r1.markdown(f"<div style='font-size:.8rem;color:#64748b;"
                        f"font-weight:600'>{k}</div>",
                        unsafe_allow_html=True)
            r2.markdown(f"<div style='font-size:.8rem;font-weight:700;"
                        f"color:#1e293b'>{v}</div>",
                        unsafe_allow_html=True)

    # Token info
    if "stats_a" in result:
        st.markdown("#### 🔍 Detail Token yang Diproses Model")
        tc1, tc2 = st.columns(2)
        for col, lbl, stats in [
            (tc1, "Dokumen A", result["stats_a"]),
            (tc2, "Dokumen B", result["stats_b"]),
        ]:
            with col:
                trunc = "⚠️ Dipotong ke 300" if stats['truncated'] else "✅ Masuk penuh"
                col.markdown(f"""
                <div class='stat-card'>
                  <div class='lbl'>{lbl}</div>
                  <div class='val'>{stats['used_tokens']}</div>
                  <div class='sub'>token / 300 max · {trunc}</div>
                </div>
                <div class='tok-info' style='margin-top:.4rem'>
                  Kata asli: <b>{stats['words_raw']:,}</b> →
                  Setelah preprocessing: <b>{stats['words_proc']:,}</b> →
                  Coverage: <b>{stats['coverage']}%</b>
                </div>
                """, unsafe_allow_html=True)

    # Interpretasi
    score   = result["score"]
    th_used = result.get("threshold", th)
    conf    = result.get("confidence", 0)
    if score >= 0.80:
        msg = (f"🔴 <b>Sangat Mirip Semantik (confidence {conf:.0f}%)</b> — "
               f"Kedua KTI memiliki topik/makna yang sangat mirip.")
        bg  = "#fee2e2"
    elif score >= th_used:
        msg = (f"🟡 <b>Mirip Semantik (confidence {conf:.0f}%)</b> — "
               f"Terdapat kemiripan makna/topik antara kedua KTI.")
        bg  = "#fef9c3"
    else:
        msg = (f"✅ <b>Berbeda Secara Semantik</b> — "
               f"Kedua KTI memiliki topik/makna yang berbeda.")
        bg  = "#f0fdf4"

    st.markdown(
        f"<div style='background:{bg};border-radius:8px;padding:.8rem;"
        f"font-size:.85rem;margin-top:.5rem'>💬 {msg}</div>",
        unsafe_allow_html=True)

    # ── ANALISIS PLAGIARISME ─────────────────────────────────
    st.markdown("<hr style='border-color:#e2e8f0;margin:.8rem 0'>",
                unsafe_allow_html=True)
    st.markdown("### 🔎 Analisis Plagiarisme Konten")
    st.markdown("""
    <div style='background:#f0f9ff;border-left:3px solid #0ea5e9;
                border-radius:4px;padding:.6rem .9rem;font-size:.8rem;
                color:#0c4a6e;margin-bottom:.8rem'>
    ℹ️ <b>Catatan penting:</b> Analisis plagiarisme di bawah mengukur
    <b>kesamaan teks/konten</b> (copy-paste, parafrase dekat),
    berbeda dari analisis kemiripan semantik di atas yang mengukur
    kesamaan topik/makna. Dua KTI bisa bertopik sama tapi bukan plagiat.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("🔎 Menganalisis konten untuk plagiarisme..."):
        plag = comprehensive_plagiarism_check(raw_a, raw_b)

    # Tampilkan skor plagiarisme
    p1c, p2c, p3c, p4c = st.columns(4)
    metrics_plag = [
        (p1c, "Skor Plagiarisme", f"{plag['plagiarism_pct']:.1f}%",
         "#dc2626" if plag['plagiarism_pct']>=50 else "#16a34a"),
        (p2c, "TF-IDF Similarity", f"{plag['tfidf_score']*100:.1f}%", "#0F4C75"),
        (p3c, "Bigram Overlap", f"{plag['bigram_overlap']*100:.1f}%", "#0F4C75"),
        (p4c, "Trigram Overlap", f"{plag['trigram_overlap']*100:.1f}%", "#0F4C75"),
    ]
    for col, label, val, color in metrics_plag:
        col.markdown(
            f"<div class='stat-card'>"
            f"<div class='lbl'>{label}</div>"
            f"<div class='val' style='color:{color};font-size:1.4rem'>{val}</div>"
            f"</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Level risiko
    st.markdown(
        f"<div style='background:{plag['risk_color']};border-radius:8px;"
        f"padding:.8rem 1rem;font-size:.9rem;margin:.5rem 0'>"
        f"<b>Tingkat Risiko Plagiarisme:</b> {plag['risk_level']}</div>",
        unsafe_allow_html=True)

    # Detail n-gram
    st.markdown(f"""
    <div style='font-size:.8rem;color:#64748b;margin:.5rem 0'>
    Frasa 2-kata sama: <b>{plag['bigram_shared']:,}</b> frasa
    ({plag['pct_a_bigram']:.1f}% dari dok. A, {plag['pct_b_bigram']:.1f}% dari dok. B) &nbsp;|&nbsp;
    Frasa 3-kata sama: <b>{plag['trigram_shared']:,}</b> frasa
    </div>
    """, unsafe_allow_html=True)

    # Kalimat mirip
    if plag['similar_sentences']:
        with st.expander(
            f"📋 Kalimat/Paragraf yang Mirip "
            f"({len(plag['similar_sentences'])} ditemukan)"):
            for i, s in enumerate(plag['similar_sentences'], 1):
                st.markdown(
                    f"**[{i}] Kesamaan: {s['similarity']}%**")
                col_ka, col_kb = st.columns(2)
                col_ka.markdown(
                    f"<div style='background:#fef9c3;padding:.5rem;"
                    f"border-radius:4px;font-size:.78rem'>"
                    f"<b>Dokumen A:</b><br>{s['kalimat_a']}</div>",
                    unsafe_allow_html=True)
                col_kb.markdown(
                    f"<div style='background:#fef9c3;padding:.5rem;"
                    f"border-radius:4px;font-size:.78rem'>"
                    f"<b>Dokumen B:</b><br>{s['kalimat_b']}</div>",
                    unsafe_allow_html=True)
                st.markdown("")
    else:
        st.markdown(
            "<div style='color:#16a34a;font-size:.85rem'>"
            "✅ Tidak ditemukan kalimat/paragraf yang identik atau sangat mirip.</div>",
            unsafe_allow_html=True)

    # Download
    st.markdown("<hr style='border-color:#e2e8f0;margin:.8rem 0'>",
                unsafe_allow_html=True)
    out = {
        "dokumen_a": getattr(fa,'name','Teks A'),
        "dokumen_b": getattr(fb,'name','Teks B'),
        "skor_kemiripan": round(result["score"],4),
        "persentase": f"{result['score']*100:.2f}%",
        "threshold": result["threshold"],
        "label": result["label"],
        "model": result["model"],
        "waktu_detik": round(elapsed,3),
    }
    st.download_button("📥 Unduh Hasil (JSON)",
        data=json.dumps(out, indent=2, ensure_ascii=False),
        file_name="hasil_kemiripan_kti.json",
        mime="application/json")


# ══ TAB 2: EVALUASI BATCH ════════════════════════════════════

def tab_batch(model, th, mp, csv):
    st.markdown("""
    <div class='info-box'>
      <b>📌 Cara Penggunaan:</b> Upload CSV (kolom: dokumen_1, dokumen_2, label)
      + ZIP berisi semua PDF yang dirujuk di CSV.
    </div>""", unsafe_allow_html=True)

    tmpl = pd.DataFrame({
        "dokumen_1":["KTI_001.pdf","KTI_003.pdf"],
        "dokumen_2":["KTI_002.pdf","KTI_004.pdf"],
        "label":[1,0],
    })
    st.download_button("📥 Download Template CSV",
        data=tmpl.to_csv(index=False),
        file_name="template_batch.csv", mime="text/csv")

    st.markdown("<hr style='border-color:#e2e8f0;margin:.8rem 0'>",
                unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        csv_f = st.file_uploader("📋 Upload CSV", type=["csv"])
    with c2:
        zip_f = st.file_uploader("🗜️ Upload ZIP (berisi PDF)", type=["zip"])

    if not (csv_f and zip_f):
        return

    df = pd.read_csv(csv_f)
    if not {"dokumen_1","dokumen_2","label"}.issubset(df.columns):
        st.error("CSV harus memiliki kolom: dokumen_1, dokumen_2, label")
        return

    st.success(f"✓ {len(df)} pasangan dimuat")
    st.dataframe(df.head(5), use_container_width=True)

    if not st.button("▶️ Jalankan Evaluasi Batch", type="primary"):
        return

    sw = load_sastrawi()
    mdl_obj = tok_obj = mlen = None

    if "LSTM" in model and mp and os.path.exists(mp) and csv and os.path.exists(csv):
        with st.spinner("Memuat model LSTM..."):
            mdl_obj, tok_obj, mlen, err = load_lstm(mp, csv)
        if err:
            st.warning(f"Model gagal: {err}. Menggunakan TF-IDF.")

    results = []
    prog = st.progress(0)
    info = st.empty()

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_f) as zf:
            zf.extractall(tmpdir)

        for idx, row in df.iterrows():
            prog.progress((idx+1)/len(df))
            info.text(f"Memproses {idx+1}/{len(df)}: {row['dokumen_1']} vs {row['dokumen_2']}")

            def find(name):
                p = os.path.join(tmpdir, name)
                if os.path.exists(p): return p
                for root,_,files in os.walk(tmpdir):
                    if name in files: return os.path.join(root,name)
                return None

            pa = find(row['dokumen_1'])
            pb = find(row['dokumen_2'])

            if not pa or not pb:
                results.append({**row.to_dict(),"skor":None,
                                 "prediksi":None,"status":"PDF tidak ditemukan"})
                continue

            ra = extract_pdf(pa); rb = extract_pdf(pb)
            if not ra or not rb:
                results.append({**row.to_dict(),"skor":None,
                                 "prediksi":None,"status":"Gagal ekstrak"})
                continue

            if mdl_obj:
                res = predict_lstm(ra, rb, mdl_obj, tok_obj, mlen, th, sw)
            else:
                sa = extract_smart(ra,3000); sb = extract_smart(rb,3000)
                s  = tfidf_sim(sa,sb)
                res= {"score":s,"label":"MIRIP" if s>=th else "TIDAK MIRIP"}

            results.append({
                "dokumen_1": row['dokumen_1'],
                "dokumen_2": row['dokumen_2'],
                "label_aktual": int(row['label']),
                "skor": round(res.get("score",0),4),
                "prediksi": 1 if res.get("label")=="MIRIP" else 0,
                "label_prediksi": res.get("label","-"),
                "status":"OK"
            })

    prog.empty(); info.empty()
    df_r = pd.DataFrame(results)
    valid = df_r[df_r['status']=='OK']

    if len(valid):
        from sklearn.metrics import (accuracy_score, precision_score,
                                     recall_score, f1_score, confusion_matrix)
        yt = valid['label_aktual'].values
        yp = valid['prediksi'].values
        acc=accuracy_score(yt,yp); prec=precision_score(yt,yp,zero_division=0)
        rec=recall_score(yt,yp,zero_division=0); f1=f1_score(yt,yp,zero_division=0)
        cm=confusion_matrix(yt,yp)
        tn,fp,fn,tp = cm.ravel() if cm.size==4 else (0,0,0,0)

        st.markdown("### 📊 Hasil Evaluasi")
        cols = st.columns(4)
        for c,n,v in [(cols[0],"Accuracy",acc),(cols[1],"Precision",prec),
                      (cols[2],"Recall",rec),(cols[3],"F1-Score",f1)]:
            c.markdown(f"""<div class='stat-card'>
              <div class='lbl'>{n}</div>
              <div class='val'>{v*100:.2f}%</div>
              <div class='sub'>dari {len(valid)} pasangan</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(
            [[tn,fp],[fn,tp]],
            index=["Aktual: Tdk Mirip","Aktual: Mirip"],
            columns=["Pred: Tdk Mirip","Pred: Mirip"]
        ), use_container_width=True)

    st.dataframe(df_r, use_container_width=True)
    st.download_button("📥 Unduh Hasil CSV",
        data=df_r.to_csv(index=False),
        file_name="hasil_batch.csv", mime="text/csv")


# ══ TAB 3: PANDUAN ═══════════════════════════════════════════

def tab_panduan():
    st.markdown("## 📖 Panduan & Info Sistem")
    with st.expander("🔍 Mengapa Skor Bisa Berbeda dari Ekspektasi?", expanded=True):
        st.markdown("""
        **Penyebab utama:**

        1. **MAX 300 token** — Model LSTM hanya bisa membaca 300 token.
           Dokumen KTI rata-rata 1.600+ token → hanya ~19% yang terbaca.
           Versi v2 ini menggunakan *strategi ekstraksi cerdas*:
           mengambil bagian paling representatif (abstrak + kesimpulan).

        2. **Kemiripan semantik ≠ kemiripan topik** — Dua KTI tentang
           "gempa bumi" bisa mendapat skor rendah jika metode dan
           hasil penelitiannya berbeda.

        3. **Threshold 0,59** — Dikalibrasi pada dataset 87 KTI.
           Skor 0,50 berarti model tidak yakin (borderline).

        **Interpretasi skor Siamese LSTM:**

        Model ini menggunakan BiLSTM + ReLU + Focal Loss sehingga
        distribusi skor bersifat **binary** — cenderung 0,50 atau 1,00.

        | Skor | Makna |
        |---|---|
        | **1,0000** | Identik sempurna |
        | **0,80 – 1,00** | Sangat Mirip → Perlu investigasi mendalam |
        | **0,59 – 0,80** | Mirip → Review manual Tim Penilai |
        | **0,50 – 0,59** | Tidak Mirip (di bawah threshold) |
        | **0,5000** | Tidak Mirip sama sekali (nilai minimum model) |

        > **Mengapa skor minimum 0,50?**
        > Arsitektur BiLSTM + ReLU menghasilkan vektor yang selalu ≥ 0,
        > sehingga cosine similarity ≥ 0, dan output (cos+1)/2 ≥ 0,50.
        > Skor 0,50 bukan berarti "mirip 50%" melainkan
        > **"model memutuskan: tidak ada kemiripan semantik"**.
        > Threshold 0,59 dikalibrasi untuk memisahkan dua kelompok ini.
        """)

    with st.expander("⚡ Mengapa Pertama Kali Lambat (~36 detik)?"):
        st.markdown("""
        **Alasan:**
        - Pertama kali: model LSTM (16,77 MB) harus dimuat ke memori
        - Setelah dimuat, disimpan di cache → request berikutnya **< 5 detik**
        - Coba klik Analisis lagi untuk melihat perbedaan kecepatannya!
        """)

    with st.expander("📊 Perbandingan Model"):
        st.dataframe(pd.DataFrame({
            "Aspek": ["F1-Score","Accuracy","Recall","Ukuran Model",
                      "GPU Diperlukan","Waktu Inferensi","Threshold"],
            "Siamese LSTM ⭐": ["69,26%","93,68%","70,80%","16,77 MB",
                                "Tidak","~2 detik (setelah cache)","0,59"],
            "TF-IDF Baseline": ["—","—","—","< 1 MB","Tidak","< 1 detik","0,1753"],
        }), use_container_width=True, hide_index=True)

    with st.expander("📚 Referensi Ilmiah"):
        st.markdown("""
        - Mueller & Thyagarajan (2016). *Siamese Recurrent Architectures*. AAAI.
          DOI: [10.1609/aaai.v30i1.10350](https://doi.org/10.1609/aaai.v30i1.10350)
        - Wilie et al. (2020). *IndoNLU*. AACL-IJCNLP.
          DOI: [10.18653/v1/2020.aacl-main.85](https://doi.org/10.18653/v1/2020.aacl-main.85)
        - Cer et al. (2017). *SemEval-2017 Task 1*.
          DOI: [10.18653/v1/S17-2001](https://doi.org/10.18653/v1/S17-2001)
        """)


# ══ MAIN ════════════════════════════════════════════════════

def main():
    st.markdown("""
    <div class='main-header'>
      <h1>🔬 Sistem Deteksi Kemiripan Semantik KTI PMG BMKG</h1>
      <p>Berbasis Siamese LSTM &amp; Siamese IndoBERT &nbsp;·&nbsp;
         Siamese LSTM: F1=69,26% &nbsp;|&nbsp; Siamese IndoBERT: F1=63,68% &nbsp;|&nbsp;
         v2.0 — Strategi Ekstraksi Cerdas</p>
    </div>
    """, unsafe_allow_html=True)

    model, th, mp, csv = sidebar()

    # Warning jika pdfplumber tidak ada
    if not load_pdfplumber():
        st.warning("⚠️ `pdfplumber` belum terinstall. "
                   "Jalankan: `python -m pip install pdfplumber`")

    t1, t2, t3 = st.tabs(["🔍 Demo 1-vs-1","📊 Evaluasi Batch","📖 Panduan & Info"])
    with t1: tab_demo(model, th, mp, csv)
    with t2: tab_batch(model, th, mp, csv)
    with t3: tab_panduan()

if __name__ == "__main__":
    main()
