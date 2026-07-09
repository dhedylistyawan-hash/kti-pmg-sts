"""
Modul prapemrosesan teks untuk KTI PMG BMKG.
Strategi: ekstrak teks representatif dari dokumen panjang
agar model LSTM (MAX 300 token) mendapat informasi terbaik.
"""

import re
from typing import Tuple


# ── Kata-kata yang menandai awal konten utama KTI ─────────────
CONTENT_START_MARKERS = [
    r'abstrak',
    r'abstract',
    r'pendahuluan',
    r'bab\s*i\b',
    r'latar\s*belakang',
    r'i\.\s*pendahuluan',
]

# ── Konten yang harus dihapus sebelum analisis ─────────────────
NOISE_PATTERNS = [
    # Daftar isi dan halaman
    r'daftar\s*isi[\s\S]{0,2000}?(?=bab\s*i|abstrak|pendahuluan)',
    r'daftar\s*tabel[\s\S]{0,500}?(?=\n\n|\Z)',
    r'daftar\s*gambar[\s\S]{0,500}?(?=\n\n|\Z)',
    r'daftar\s*lampiran[\s\S]{0,500}?(?=\n\n|\Z)',
    # Nomor halaman
    r'\b(?:halaman|hal|page)\s*\d+\b',
    r'^\s*\d+\s*$',
    # Header/footer berulang
    r'badan\s*meteorologi[\s\S]{0,100}?geofisika',
    r'bmkg[\s\S]{0,50}?(?:\d{4})',
    # Titik-titik (daftar isi)
    r'\.{3,}\s*\d+',
    # Karakter sampah
    r'[^\w\s\.,;:\!\?\-\(\)\/\%]',
]

# ── Bagian terpenting KTI untuk representasi ─────────────────
IMPORTANT_SECTIONS = [
    r'abstrak',
    r'abstract',
    r'latar\s*belakang',
    r'tujuan\s*penelitian',
    r'metodologi',
    r'metode\s*penelitian',
    r'hasil\s*(?:dan\s*)?pembahasan',
    r'kesimpulan',
]


def clean_text(text: str) -> str:
    """Bersihkan teks dari noise umum PDF."""
    if not text:
        return ""

    # Normalisasi whitespace
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\t', ' ', text)

    # Hapus baris yang hanya angka (nomor halaman)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # Hapus titik-titik daftar isi
    text = re.sub(r'\.{3,}\s*\d+', '', text)

    # Hapus karakter non-ASCII yang umum dari scan PDF
    text = re.sub(r'[^\x00-\x7F\u00C0-\u024F]', ' ', text)

    # Hapus spasi berlebihan
    text = re.sub(r' {3,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def extract_main_content(text: str,
                         max_chars: int = 8000) -> str:
    """
    Ekstrak konten utama KTI dengan strategi hierarkis:
    1. Coba ekstrak mulai dari Abstrak/BAB I
    2. Jika tidak ditemukan, ambil dari awal (skip 500 char pertama)
    3. Batasi ke max_chars karakter
    """
    text_lower = text.lower()

    # Cari posisi awal konten utama
    start_pos = 0
    for marker in CONTENT_START_MARKERS:
        match = re.search(marker, text_lower)
        if match:
            # Mulai dari 50 char sebelum marker untuk konteks
            start_pos = max(0, match.start() - 50)
            break

    # Jika tidak ketemu marker, skip bagian awal (judul, cover)
    if start_pos == 0:
        # Skip 10% pertama (biasanya cover, daftar isi)
        start_pos = max(0, len(text) // 10)

    content = text[start_pos:]

    # Batasi panjang
    if len(content) > max_chars:
        content = content[:max_chars]

    return content.strip()


def extract_representative_text(text: str,
                                 target_tokens: int = 280) -> str:
    """
    Strategi TERBAIK untuk MAX 300 token:
    Ambil bagian paling representatif dari seluruh dokumen.

    Pendekatan: 60% awal + 40% akhir konten utama
    Karena:
    - Awal (Abstrak + Pendahuluan) = konteks dan tujuan
    - Akhir (Kesimpulan) = hasil dan kontribusi
    - Keduanya sangat representatif untuk STS
    """
    # Estimasi: 1 token ≈ 5-6 karakter setelah preprocessing
    target_chars = target_tokens * 5

    content = extract_main_content(text, max_chars=target_chars * 3)

    if len(content) <= target_chars:
        return content

    # Ambil 60% dari awal + 40% dari akhir
    part_a_len = int(target_chars * 0.60)
    part_b_len = int(target_chars * 0.40)

    part_a = content[:part_a_len]
    part_b = content[-part_b_len:] if len(content) > part_b_len else ""

    # Sambungkan dengan penanda
    if part_b:
        combined = part_a + " ... " + part_b
    else:
        combined = part_a

    return combined.strip()


def preprocess_for_lstm(text: str,
                         stopword_remover=None) -> str:
    """Pipeline preprocessing untuk Siamese LSTM."""
    if not text:
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Hapus angka
    text = re.sub(r'\d+', ' ', text)

    # 3. Hapus tanda baca dan karakter non-huruf
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # 4. Normalisasi spasi
    text = re.sub(r'\s+', ' ', text).strip()

    # 5. Hapus stopword (PySastrawi)
    if stopword_remover:
        try:
            text = stopword_remover.remove(text)
        except Exception:
            pass

    return text


def preprocess_for_bert(text: str) -> str:
    """Pipeline preprocessing untuk Siamese IndoBERT (tanpa stopword)."""
    if not text:
        return ""

    # Hanya normalisasi ringan, pertahankan konteks
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_text_stats(raw_text: str,
                   processed_text: str) -> dict:
    """Hitung statistik teks untuk ditampilkan ke pengguna."""
    raw_words  = len(raw_text.split())
    proc_words = len(processed_text.split())

    # Estimasi token (1 kata ≈ 1.2 token rata-rata)
    est_tokens = int(proc_words * 1.0)
    final_tokens = min(est_tokens, 300)

    coverage_pct = min(100, round(300 / max(est_tokens, 1) * 100))

    return {
        'raw_words'      : raw_words,
        'raw_chars'      : len(raw_text),
        'processed_words': proc_words,
        'estimated_tokens': est_tokens,
        'final_tokens'   : final_tokens,
        'model_coverage' : f"{coverage_pct}%",
        'is_truncated'   : est_tokens > 300,
    }
