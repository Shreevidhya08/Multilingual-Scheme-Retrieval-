# search_engine.py
# ─────────────────────────────────────────────────────────────────────────────
# Multilingual Government Scheme Search Engine
# Supports: English, Hindi (हिंदी), Kannada (ಕನ್ನಡ)
# Search modes: Fuzzy name match + Semantic FAISS + BM25 (hybrid)
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import subprocess
import warnings

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL     = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K           = 5
SEMANTIC_WEIGHT = 0.7
BM25_WEIGHT     = 0.3
BASE_URL        = "https://www.myscheme.gov.in/schemes/"


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe(val) -> str:
    """Convert cell value to string; return empty string for NaN."""
    # BUG FIX 3: unified safe() that also catches float NaN passed as object
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val).strip()


def tokenize(text: str) -> list:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\w+", str(text).lower())


def detect_language(text: str) -> str:
    """
    Detect language from Unicode script — no library needed.
    Kannada: U+0C80–U+0CFF  |  Hindi: U+0900–U+097F  |  else: English
    """
    if re.search(r"[\u0C80-\u0CFF]", text):
        return "kn"
    elif re.search(r"[\u0900-\u097F]", text):
        return "hi"
    return "en"


def build_search_text(row) -> str:
    """
    Concatenates English + Hindi + Kannada fields so a query in ANY
    language can match against this single string.
    """
    parts = [
        safe(row.get("scheme_name",         "")),
        safe(row.get("scheme_name_hindi",   "")),
        safe(row.get("scheme_name_kannada", "")),
        safe(row.get("tags",    "")),
        safe(row.get("tags_hi", "")),
        safe(row.get("tags_kn", "")),
        safe(row.get("schemeCategory", "")),
        safe(row.get("level",          "")),
        safe(row.get("details",    ""))[:400],
        safe(row.get("details_hi", ""))[:200],
        safe(row.get("eligibility", ""))[:200],
        safe(row.get("benefits",    ""))[:200],
    ]
    return " | ".join(p for p in parts if p)


# ── Engine loader ─────────────────────────────────────────────────────────────

def load_engine(
    csv_path: str,
    embeddings_path: str = "scheme_embeddings.npy",
    index_path: str      = "scheme_faiss.index",
):
    """
    Load (or build) all search artefacts and return them as a tuple.

    BUG FIX 1: Always rebuilds embeddings + FAISS index from scratch if
    the downloaded files exist but were built from a different CSV version.
    We do this by storing a checksum of the CSV alongside the embeddings,
    and regenerating whenever the checksum doesn't match.

    Returns
    -------
    df          : pd.DataFrame  — scheme data with search_text column
    embed_model : SentenceTransformer
    index       : faiss.Index
    bm25        : BM25Okapi
    """
    import hashlib

    CHECKSUM_PATH = embeddings_path + ".csv_md5"

    # ── 1. Load CSV ───────────────────────────────────────────────────────────
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    df.reset_index(drop=True, inplace=True)
    num_rows = len(df)
    print(f"  ✅ {num_rows:,} rows, {len(df.columns)} columns")

    df["search_text"] = df.apply(build_search_text, axis=1)
    print("  ✅ search_text column built")

    # Compute CSV checksum to detect stale artefacts
    with open(csv_path, "rb") as f:
        csv_md5 = hashlib.md5(f.read()).hexdigest()

    def _artefacts_are_fresh():
        """Return True only if both files exist, row counts match, and CSV hasn't changed."""
        if not os.path.exists(embeddings_path) or not os.path.exists(index_path):
            return False
        if not os.path.exists(CHECKSUM_PATH):
            return False
        with open(CHECKSUM_PATH) as f:
            saved_md5 = f.read().strip()
        if saved_md5 != csv_md5:
            print("  ⚠️  CSV has changed since embeddings were built — rebuilding…")
            return False
        emb = np.load(embeddings_path)
        if emb.shape[0] != num_rows:
            print(f"  ⚠️  Row count mismatch (embeddings={emb.shape[0]}, CSV={num_rows}) — rebuilding…")
            return False
        idx = faiss.read_index(index_path)
        if idx.ntotal != num_rows:
            print(f"  ⚠️  Row count mismatch (FAISS={idx.ntotal}, CSV={num_rows}) — rebuilding…")
            return False
        return True

    # ── 2. Sentence-Transformer ───────────────────────────────────────────────
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    embed_model = SentenceTransformer(EMBED_MODEL)
    print("  ✅ Embedding model ready")

    # ── 3. Embeddings ─────────────────────────────────────────────────────────
    if _artefacts_are_fresh():
        print(f"\nLoading saved embeddings: {embeddings_path}")
        embeddings = np.load(embeddings_path)
        print(f"  ✅ Embeddings ready — shape: {embeddings.shape}")
        print(f"\nLoading saved FAISS index: {index_path}")
        index = faiss.read_index(index_path)
        print(f"  ✅ FAISS index ready — {index.ntotal:,} vectors")
    else:
        # Remove stale files
        for path in [embeddings_path, index_path, CHECKSUM_PATH]:
            if os.path.exists(path):
                os.remove(path)

        print(f"\nGenerating embeddings for {num_rows:,} rows (~2–5 min on CPU)…")
        embeddings = embed_model.encode(
            df["search_text"].tolist(),
            batch_size           = 64,
            show_progress_bar    = True,
            convert_to_numpy     = True,
            normalize_embeddings = True,
        )
        np.save(embeddings_path, embeddings)
        print(f"  ✅ Embeddings saved — shape: {embeddings.shape}")

        print("\nBuilding FAISS index…")
        dim   = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))
        faiss.write_index(index, index_path)
        print(f"  ✅ FAISS index saved — {index.ntotal:,} vectors")

        # Save checksum so next run knows these artefacts match this CSV
        with open(CHECKSUM_PATH, "w") as f:
            f.write(csv_md5)

    # ── 4. BM25 index ─────────────────────────────────────────────────────────
    print("\nBuilding BM25 index…")
    tokenized_corpus = [tokenize(t) for t in df["search_text"]]
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"  ✅ BM25 index ready — {len(tokenized_corpus):,} documents")

    return df, embed_model, index, bm25


# ── Search functions ──────────────────────────────────────────────────────────

def fuzzy_name_match(
    query: str,
    df: pd.DataFrame,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """
    Fuzzy match query against the scheme_name column.
    Catches misspellings like 'pradan' → 'Pradhan'.
    """
    query_clean  = query.strip().lower()
    num_rows     = len(df)
    scheme_names = df["scheme_name"].fillna("").tolist()

    matches = process.extract(
        query_clean,
        scheme_names,
        scorer = fuzz.token_sort_ratio,
        limit  = top_k * 2,
    )

    results, seen = [], set()
    for _match_text, score, idx in matches:
        # BUG FIX 2: guard against out-of-bound indices from rapidfuzz
        if idx in seen or score < 40 or idx >= num_rows:
            continue
        seen.add(idx)
        row = df.iloc[idx]
        results.append({
            "idx":                 idx,
            "fuzzy_score":         score,
            "scheme_name":         safe(row.get("scheme_name",         "")),
            "scheme_name_hindi":   safe(row.get("scheme_name_hindi",   "")),
            "scheme_name_kannada": safe(row.get("scheme_name_kannada", "")),
            "tags":                safe(row.get("tags",   "")),
            "level":               safe(row.get("level",  "")),
            "schemeCategory":      safe(row.get("schemeCategory", "")),
            "slug":                safe(row.get("slug",   "")),
            "details_snippet":     safe(row.get("details", ""))[:250] + "...",
        })

    return pd.DataFrame(results)


def hybrid_search(
    query:       str,
    df:          pd.DataFrame,
    embed_model: SentenceTransformer,
    index:       faiss.Index,
    bm25:        BM25Okapi,
    top_k:       int = TOP_K,
) -> pd.DataFrame:
    """
    Three-stage hybrid search:
      1. Fuzzy name match   (catches misspellings)
      2. Semantic FAISS     (meaning-based, multilingual)
      3. BM25 keyword       (exact-token recall)

    Results are de-duplicated and sorted by similarity score.
    """
    query_clean  = query.strip()
    seen_indices = set()
    results      = []
    num_rows     = len(df)

    # ── Stage 1: Fuzzy ────────────────────────────────────────────────────────
    fuzzy_df = fuzzy_name_match(query_clean, df, top_k=top_k)

    for _, row in fuzzy_df.iterrows():
        idx = int(row["idx"])
        if idx >= num_rows:
            continue
        seen_indices.add(idx)
        results.append({
            "rank":                None,
            "similarity":          round(row["fuzzy_score"] / 100, 4),
            "match_type":          f"fuzzy ({row['fuzzy_score']}%)",
            "scheme_name":         row["scheme_name"],
            "scheme_name_hindi":   row["scheme_name_hindi"],
            "scheme_name_kannada": row["scheme_name_kannada"],
            "tags":                row["tags"],
            "level":               row["level"],
            "schemeCategory":      row["schemeCategory"],
            "slug":                row["slug"],
            "details_snippet":     row["details_snippet"],
        })

    # ── Stage 2: Semantic (FAISS) ─────────────────────────────────────────────
    query_embedding = embed_model.encode(
        [query_clean],
        convert_to_numpy     = True,
        normalize_embeddings = True,
    ).astype(np.float32)

    scores, faiss_indices = index.search(query_embedding, top_k * 2)

    for score, idx in zip(scores[0], faiss_indices[0]):
        idx = int(idx)
        if idx < 0 or idx >= num_rows or idx in seen_indices:
            continue
        seen_indices.add(idx)
        row = df.iloc[idx]
        results.append({
            "rank":                None,
            "similarity":          round(float(score), 4),
            "match_type":          "semantic",
            # BUG FIX 3: all fields go through safe() — no raw .get() without NaN guard
            "scheme_name":         safe(row.get("scheme_name",         "")),
            "scheme_name_hindi":   safe(row.get("scheme_name_hindi",   "")),
            "scheme_name_kannada": safe(row.get("scheme_name_kannada", "")),
            "tags":                safe(row.get("tags",   "")),
            "level":               safe(row.get("level",  "")),
            "schemeCategory":      safe(row.get("schemeCategory", "")),
            "slug":                safe(row.get("slug",   "")),
            "details_snippet":     safe(row.get("details", ""))[:250] + "...",
        })

    # ── Stage 3: BM25 keyword ─────────────────────────────────────────────────
    bm25_scores = bm25.get_scores(tokenize(query_clean))
    top_bm25    = np.argsort(bm25_scores)[::-1][: top_k * 2]

    for idx in top_bm25:
        idx = int(idx)
        if idx >= num_rows or idx in seen_indices or bm25_scores[idx] == 0:
            continue
        seen_indices.add(idx)
        row = df.iloc[idx]
        norm_score = round(float(bm25_scores[idx]) / (float(bm25_scores[top_bm25[0]]) + 1e-9), 4)
        results.append({
            "rank":                None,
            "similarity":          norm_score,
            "match_type":          "bm25",
            "scheme_name":         safe(row.get("scheme_name",         "")),
            "scheme_name_hindi":   safe(row.get("scheme_name_hindi",   "")),
            "scheme_name_kannada": safe(row.get("scheme_name_kannada", "")),
            "tags":                safe(row.get("tags",   "")),
            "level":               safe(row.get("level",  "")),
            "schemeCategory":      safe(row.get("schemeCategory", "")),
            "slug":                safe(row.get("slug",   "")),
            "details_snippet":     safe(row.get("details", ""))[:250] + "...",
        })

    # ── Merge, sort, rank ─────────────────────────────────────────────────────
    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    result_df = (
        result_df
        .sort_values("similarity", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    result_df["rank"] = result_df.index + 1

    return result_df


# ── Voice / STT helpers ───────────────────────────────────────────────────────

LANG_DISPLAY = {
    "hi": "Hindi / हिंदी",
    "kn": "Kannada / ಕನ್ನಡ",
    "en": "English",
    "te": "Telugu",
    "ta": "Tamil",
    "mr": "Marathi",
}


def convert_to_wav(input_path: str, output_path: str = "query_converted.wav") -> str:
    """Convert any audio format to 16 kHz mono WAV for Whisper."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed:\n{result.stderr}")
    return output_path


def transcribe_audio(
    stt_model,
    audio_path: str,
    language: str = None,
) -> tuple:
    """Transcribe audio to text using Whisper."""
    wav_path = convert_to_wav(audio_path)

    options = {
        "task":                       "transcribe",
        "beam_size":                  5,
        "best_of":                    5,
        "temperature":                0.0,
        "condition_on_previous_text": False,
        "fp16":                       False,
    }
    if language:
        options["language"] = language

    result   = stt_model.transcribe(wav_path, **options)
    text     = result["text"].strip()
    detected = result.get("language", "unknown")

    if os.path.exists(wav_path):
        os.remove(wav_path)

    return text, detected
