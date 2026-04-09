# app.py
# ─────────────────────────────────────────────────────────────────────────────
# Multilingual Government Scheme Search — Streamlit Cloud compatible
# Fixes: replaced sounddevice + scipy (PortAudio) with st.audio_input
# Adds:  @st.cache_resource for heavy models (engine + Whisper)
# ─────────────────────────────────────────────────────────────────────────────

import os
import tempfile

import whisper
import streamlit as st

from search_engine import load_engine, hybrid_search, detect_language

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Scheme Search",
    page_icon  = "🏛️",
    layout     = "wide",
)

# ── Cached loaders (runs once per session / worker) ───────────────────────────

@st.cache_resource(show_spinner="Loading search engine…")
def get_engine():
    """Load CSV, embeddings, FAISS index and BM25 — cached across reruns."""
    return load_engine(
        "data/finalized_data.csv",
        "scheme_embeddings.npy",
        "scheme_faiss.index",
    )


@st.cache_resource(show_spinner="Loading speech-to-text model…")
def get_stt_model():
    """Load Whisper model once and reuse — avoids reloading on every interaction.

    Use 'small' or 'base' on Streamlit Cloud free tier (1 GB RAM limit).
    Switch to 'medium' if you have a paid instance with more memory.
    """
    return whisper.load_model("small")   # change to "medium" on paid tier


# ── Load engine ───────────────────────────────────────────────────────────────
df, embed_model, index, bm25 = get_engine()

# ── Language labels ───────────────────────────────────────────────────────────
LANG_LABEL = {
    "en": "English 🇬🇧",
    "hi": "Hindi 🇮🇳",
    "kn": "Kannada 🇮🇳",
}

BASE_URL = "https://www.myscheme.gov.in/schemes/"

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏛️ Government Scheme Search")
st.caption("Search in English, Hindi (हिंदी), or Kannada (ಕನ್ನಡ)")

# Initialise query in session state so voice can set it
if "query" not in st.session_state:
    st.session_state.query = ""

# ── Text input ────────────────────────────────────────────────────────────────
text_query = st.text_input(
    "🔍 Type your query",
    value       = st.session_state.query,
    placeholder = "e.g. Pradhan Mantri Awas Yojana / किसान सम्मान निधि / ಆಯುಷ್ಮಾನ್ ಭಾರತ",
)

st.divider()

# ── Voice input (no PortAudio dependency) ────────────────────────────────────
st.markdown("**Or use voice search:**")
audio_file = st.audio_input("🎙️ Record your query")

voice_query = ""
if audio_file is not None:
    with st.spinner("Transcribing audio…"):
        # Write uploaded bytes to a temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_file.getvalue())
            tmp_path = tmp.name

        try:
            stt    = get_stt_model()
            result = stt.transcribe(tmp_path)
            voice_query = result["text"].strip()
            st.success(f"Heard: **{voice_query}**")
        except Exception as e:
            st.error(f"Transcription failed: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

# ── Resolve active query (voice takes priority when present) ──────────────────
query = voice_query if voice_query else text_query.strip()

# ── Results ───────────────────────────────────────────────────────────────────
if query:
    lang       = detect_language(query)
    lang_label = LANG_LABEL.get(lang, f"Unknown ({lang})")
    st.info(f"Detected language: {lang_label}")

    with st.spinner("Searching…"):
        results = hybrid_search(query, df, embed_model, index, bm25)

    if results.empty:
        st.warning("No schemes found. Try a different query.")
    else:
        st.markdown(f"### Top {len(results)} results for: *{query}*")

        for _, row in results.iterrows():
            rank        = row.get("rank", "—")
            name        = row.get("scheme_name", "Unknown Scheme")
            name_hi     = row.get("scheme_name_hindi",   "")
            name_kn     = row.get("scheme_name_kannada", "")
            score       = row.get("similarity", 0.0)
            match_type  = row.get("match_type", "")
            tags        = row.get("tags", "")
            level       = row.get("level", "")
            category    = row.get("schemeCategory", "")
            slug        = row.get("slug", "")
            snippet     = row.get("details_snippet", "")

            # Build scheme URL if slug exists
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
                        # Show first 5 tags as pills
                        tag_list = [t.strip() for t in tags.split(",")][:5]
                        st.markdown("🏷️ **Tags:** " + " · ".join(tag_list))

                if scheme_url:
                    st.link_button("View Scheme →", scheme_url)
