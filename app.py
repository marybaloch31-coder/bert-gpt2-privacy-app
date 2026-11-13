import html
import re

import streamlit as st
from transformers import pipeline

# =====================================================================
#  SETTINGS - change the GitHub username (Hugging Face names are set)
# =====================================================================
BERT_REPO = "Maryam033/bert-privacy-detector"
GPT_REPO = "Maryam033/gpt2-text-generation"
GITHUB_URL = "https://github.com/marybaloch31-coder/bert-gpt2-privacy-app"
# =====================================================================

st.set_page_config(page_title="BERT & GPT-2 Studio", page_icon="🔒", layout="centered")

st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 820px;}
.hero {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 50%, #a855f7 100%);
    color: #ffffff; padding: 1.8rem 1.6rem; border-radius: 18px; margin-bottom: 1.2rem;
    box-shadow: 0 10px 30px rgba(99, 102, 241, 0.25);
}
.hero h1 {color: #ffffff; margin: 0 0 .3rem 0; font-size: 1.9rem; line-height: 1.25;}
.hero p {color: rgba(255,255,255,.92); margin: 0; font-size: 1rem;}
.pill {
    display: inline-block; padding: .15rem .7rem; border-radius: 999px; font-size: .8rem;
    background: rgba(255,255,255,.2); color: #fff; margin: .6rem .4rem 0 0;
}
.badge {
    display: inline-block; padding: .1rem .6rem; border-radius: 999px; font-size: .8rem;
    background: rgba(99,102,241,.15); color: #6366f1; border: 1px solid rgba(99,102,241,.35);
}
.result {border-radius: 14px; padding: 1.1rem 1.3rem; margin: 1rem 0 .6rem 0;}
.result h2 {margin: 0 0 .2rem 0; font-size: 1.5rem;}
.result p {margin: 0; opacity: .85;}
.result-red {background: rgba(239,68,68,.12); border-left: 6px solid #ef4444;}
.result-green {background: rgba(34,197,94,.12); border-left: 6px solid #22c55e;}
.msgbox {
    border: 1px dashed rgba(128,128,128,.5); border-radius: 12px; padding: .7rem 1rem;
    margin: .4rem 0 .8rem 0; font-size: 1.05rem;
}
.gen {
    background: rgba(99,102,241,.08); border: 1px solid rgba(99,102,241,.3);
    border-radius: 14px; padding: 1.1rem 1.3rem; line-height: 1.75; font-size: 1.05rem;
}
.gen .prompt {color: #6366f1; font-weight: 700;}
.footer {text-align: center; opacity: .6; font-size: .85rem; margin-top: 2rem;}
</style>
""",
    unsafe_allow_html=True,
)

URDU_RE = re.compile(r"[\u0600-\u06FF]")
ROMAN_URDU_WORDS = {
    "hai", "hain", "mera", "meri", "mujhe", "kal", "aaj", "nahi", "kya", "aap",
    "mein", "baje", "raha", "rahi", "hoon", "karo", "kar", "ka", "ki", "shukriya",
}


def detect_language(text):
    if URDU_RE.search(text):
        return "Urdu 🇵🇰"
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    if len(words & ROMAN_URDU_WORDS) >= 2:
        return "Roman Urdu 🇵🇰"
    return "English 🇬🇧"


def is_sensitive(label):
    # works with "Sensitive" / "Non-Sensitive" and with LABEL_0 / LABEL_1
    return str(label).strip().lower() in {"sensitive", "label_1", "1"}


# max_entries=1 keeps only ONE model in memory at a time (saves RAM on free hosting)
@st.cache_resource(show_spinner=False, max_entries=1)
def load_model(task, repo):
    return pipeline(task, model=repo)


# ------------------------------- Sidebar -------------------------------

# ------------------------------- Sidebar -------------------------------
with st.sidebar:
    st.header("📌 About")

    st.write(
        "A class project that compares two Transformer types:\n\n"
        "- **BERT** (encoder-only): understands text\n"
        "- **GPT-2** (decoder-only): generates text"
    )

    st.markdown(
        f"""
        **Models**

        - [BERT Privacy Detector](https://huggingface.co/{BERT_REPO})
        - [GPT-2 Text Generation](https://huggingface.co/{GPT_REPO})

        **Code**

        - [GitHub Repository]({GITHUB_URL})
        """
    )

    st.warning(
        "Please use **fake sample data** only. "
        "Do not type your real CNIC, phone number, password, "
        "card number or OTP into this public demo."
    )

