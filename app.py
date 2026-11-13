import html
import re

import streamlit as st
from transformers import pipeline

# --------------------------- Model settings ---------------------------
BERT_REPO = "Maryam033/bert-privacy-detector"
GPT_REPO = "Maryam033/gpt2-text-generation"
GITHUB_URL = "https://github.com/marybaloch31-coder/bert-gpt2-privacy-app"

st.set_page_config(
    page_title="BERT & GPT-2 Studio",
    page_icon="🔒",
    layout="wide",
)

# ------------------------------ Styling ------------------------------
st.markdown(
    """
    <style>
    .block-container {max-width: 1100px; padding-top: 2rem; padding-bottom: 3rem;}
    .hero {
        padding: 1.5rem 1.7rem; border-radius: 16px; margin-bottom: 1.2rem;
        background: linear-gradient(120deg, #075985, #4338ca, #7e22ce);
        color: white;
    }
    .hero h1 {color: white; margin: 0 0 .4rem 0;}
    .hero p {color: #f1f5f9; margin: 0;}
    .result {
        padding: 1rem 1.2rem; border-radius: 12px; margin: .8rem 0;
        border-left: 5px solid;
    }
    .safe {background: rgba(34,197,94,.12); border-color: #22c55e;}
    .sensitive {background: rgba(239,68,68,.12); border-color: #ef4444;}
    .generated {
        background: rgba(99,102,241,.10); border: 1px solid rgba(99,102,241,.35);
        border-radius: 12px; padding: 1rem; line-height: 1.7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------ Helpers ------------------------------
URDU_RE = re.compile(r"[\u0600-\u06FF]")
ROMAN_URDU_WORDS = {
    "hai", "hain", "mera", "meri", "mujhe", "kal", "aaj", "nahi", "kya",
    "aap", "mein", "baje", "raha", "rahi", "hoon", "karo", "kar", "ka",
    "ki", "shukriya",
}

def detect_language(text):
    if URDU_RE.search(text):
        return "Urdu"
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    if len(words & ROMAN_URDU_WORDS) >= 2:
        return "Roman Urdu"
    return "English"

def label_is_sensitive(label):
    value = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    return value in {"sensitive", "label_1", "1", "class_1"}

@st.cache_resource(show_spinner=False, max_entries=2)
def load_classifier(repo):
    return pipeline(
        "text-classification",
        model=repo,
        tokenizer=repo,
    )

@st.cache_resource(show_spinner=False, max_entries=2)
def load_generator(repo):
    return pipeline(
        "text-generation",
        model=repo,
        tokenizer=repo,
    )

def show_model_error(name, error):
    st.error(f"{name} could not be loaded. Check that the Hugging Face repository exists and is public.")
    st.caption(f"Technical detail: {error}")

# ------------------------------- Sidebar -------------------------------
with st.sidebar:
    st.header("📌 About")
    st.write(
        "A class project demonstrating two Transformer models:\n\n"
        "- **BERT:** detects privacy-sensitive text.\n"
        "- **GPT-2:** continues a text prompt."
    )
    st.markdown("### Model links")
    st.markdown(f"- [BERT Privacy Detector](https://huggingface.co/{BERT_REPO})")
    st.markdown(f"- [GPT-2 Text Generation](https://huggingface.co/{GPT_REPO})")
    st.markdown("### Source code")
