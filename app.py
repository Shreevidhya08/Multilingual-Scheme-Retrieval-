# app.py
# ─────────────────────────────────────────────────────────────────────────────
# Multilingual Government Scheme Search — Streamlit Cloud compatible
# Fixes:
#   1. PortAudio error  → replaced sounddevice/scipy with st.audio_input
#   2. EmptyDataError   → downloads CSV from HuggingFace if missing/stub
#   3. IndexError       → mismatch check rebuilds embeddings/index if stale
#   4. HTTPError        → proper error handling with clear messages
#   5. Performance      → @st.cache_resource on all heavy models
# ─────────────────────────────────────────────────────────────────────────────

import os
import tempfile
import urllib.request
import urllib.error

import streamlit as st
import whisper

from search_engine import load_engine, hybrid_search, detect_language

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Scheme Search",
    page_icon  = "🏛️",
    layout     = "wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
HF_BASE  = "https://huggingface.co/datasets/Shree0804/Multilingual_Government_Schemes_Dataset/resolve/main"
BASE_URL = "https://www.myscheme.gov.in/schemes/"

LANG_LABEL = {
    "en": "English 🇬🇧",
    "hi": "Hindi 🇮🇳",
    "kn": "Kannada 🇮🇳",
}

# ── File download helper ──────────────────────────────────────────────────────

def _download_if_needed(local_path: str, url: str, label: str) -> bool:
    """
    Download file from HuggingFace only if:
      - it doesn't exist locally, OR
      - it's a Git LFS pointer stub (< 1 KB)

    Returns True on success, False on failure.
    """
    needs_download = (
        not os.path.exists(local_path)
        or os.path.getsize(local_path) < 1024
    )

    if not needs_download:
        return True

    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

    try:
        with st.status(f"⬇️ Downloading {label}…", expanded=False):
            urllib.request.urlretrieve(url, local_path)
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            st.write(f"✅ {label} — {size_mb:.1f} MB")
        return True

    except urllib.error.HTTPError as e:
        st.error(
            f"❌ Failed to download **{label}** (HTTP {e.code}).\n\n"
            f"Check that your HuggingFace dataset is set to **Public**.\n\n"
            f"URL tried: `{url}`"
        )
        return False

    except Exception as e:
        st.error(f"❌ Unexpected error downloading {label}: {e}")
        return False


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Setting up search engine…")
def get_engine():
    """
    1. Download CSV + embeddings + FAISS index from HuggingFace if needed.
    2. Pass to load_engine() which auto-rebuilds any mismatched artefacts.
    """
    files = {
        "data/finalized_data.csv": (f"{HF_BASE}/finalized_data.csv",   "Dataset CSV"),
        "scheme_embeddings.npy":   (f"{HF_BASE}/scheme_embeddings.npy", "Embeddings"),
        "scheme_faiss.index":      (f"{HF_BASE}/scheme_faiss.index",    "FAISS Index"),
    }

    for local_path, (url, label) in files.items():
        ok = _download_if_needed(local_path, url, label)
        if not ok:
            st.stop()

    return load_engine(
        "data/finalized_data.csv",
        "scheme_embeddings.npy",
        "scheme_faiss.index",
    )


@st.cache_resource(show_spinner="Loading speech model…")
def get_stt_model():
    """
    Load Whisper once and reuse across reruns.
    'small' fits in Streamlit Cloud free tier (1 GB RAM).
    Change to 'medium' on a paid/larger instance.
    """
    return whisper.load_model("small")


# ── Boot ──────────────────────────────────────────────────────────────────────
df, embed_model, index, bm25 = get_engine()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏛️ Government Scheme Search")
st.caption("Search in English, Hindi (हिंदी), or Kannada (ಕನ್ನಡ)")

# Session state for query persistence
if "query" not in st.session_state:
    st.session_state.query = ""

# ── Text input ────────────────────────────────────────────────────────────────
text_query = st.text_input(
    "🔍 Type your query",
    value       = st.session_state.query,
    placeholder = "e.g. Pradhan Mantri Awas Yojana / किसान सम्मान निधि / ಆಯುಷ್ಮಾನ್ ಭಾರತ",
)

st.divider()

# ── Voice input ───────────────────────────────────────────────────────────────
st.markdown("**Or use voice search:**")
audio_file  = st.audio_input("🎙️ Record your query")
voice_query = ""

if audio_file is not None:
    with st.spinner("Transcribing…"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_file.getvalue())
            tmp_path = tmp.name
        try:
            stt         = get_stt_model()
            result      = stt.transcribe(tmp_path)
            voice_query = result["text"].strip()
            st.success(f"Heard: **{voice_query}**")
            st.session_state.query = voice_query
        except Exception as e:
            st.error(f"Transcription failed: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

# ── Resolve active query ──────────────────────────────────────────────────────
query = voice_query if voice_query else text_query.strip()

# ── Results ───────────────────────────────────────────────────────────────────
if query:
    lang  = detect_language(query)
    label = LANG_LABEL.get(lang, f"Unknown ({lang})")
    st.info(f"Detected language: {label}")

    with st.spinner("Searching…"):
        results = hybrid_search(query, df, embed_model, index, bm25)

    if results is None or results.empty:
        st.warning("No schemes found. Try a different query.")
    else:
        st.markdown(f"### Top {len(results)} results for: *{query}*")

        for _, row in results.iterrows():
            rank       = row.get("rank", "—")
            name       = row.get("scheme_name", "Unknown Scheme")
            name_hi    = row.get("scheme_name_hindi",   "")
            name_kn    = row.get("scheme_name_kannada", "")
            score      = float(row.get("similarity", 0.0))
            match_type = row.get("match_type", "")
            tags       = row.get("tags", "")
            level      = row.get("level", "")
            category   = row.get("schemeCategory", "")
            slug       = row.get("slug", "")
            snippet    = row.get("details_snippet", "")
            scheme_url = f"{BASE_URL}{slug}" if slug else None

            header = f"**#{rank} — {name}** &nbsp; `{score:.2f}` &nbsp; *{match_type}*"

            with st.expander(header, expanded=(rank == 1)):
                col1, col2 = st.columns([2, 1])

                with col1:
                    if name_hi:
                        st.markdown(f"🇮🇳 **Hindi:** {name_hi}")
                    if name_kn:
                        st.markdown(f"🇮🇳 **Kannada:** {name_kn}")
                    if snippet:
                        st.markdown(f"📄 **Details:** {snippet}")

                with col2:
                    if level:
                        st.markdown(f"🏢 **Level:** {level}")
                    if category:
                        st.markdown(f"📂 **Category:** {category}")
                    if tags:
                        tag_list = [t.strip() for t in tags.split(",")][:5]
                        st.markdown("🏷️ **Tags:** " + " · ".join(tag_list))

                if scheme_url:
                    st.link_button("View Scheme →", scheme_url)
