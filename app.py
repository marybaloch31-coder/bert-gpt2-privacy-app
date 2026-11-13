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
with st.sidebar:
    st.header("📌 About")
    st.write(
        "A class project that compares two Transformer types:\n\n"
        "- **BERT** (encoder-only): understands text\n"
        "- **GPT-2** (decoder-only): generates text"
    )
    st.markdown(
        f"**Models**\n\n- [BERT privacy detector](https://huggingface.co/{Maryam033/bert-privacy-detector})\n"
        f"- [GPT-2 text generation](https://huggingface.co/{Maryam033/gpt2-text-generation})\n\n"
        f"**Code**\n\n- [GitHub repository]({GITHUB_URL})"
    )
    st.warning("Please use **fake sample data** only. Do not type your real CNIC, phone number, "
               "password, card number or OTP into this public demo.")

# -------------------------------- Header --------------------------------
st.markdown(
    """
<div class="hero">
  <h1>🔒 BERT &amp; GPT-2 Studio</h1>
  <p>Detect privacy-sensitive text in Urdu, English and Roman Urdu, and generate text with GPT-2.</p>
  <span class="pill">Encoder-only: BERT</span>
  <span class="pill">Decoder-only: GPT-2</span>
  <span class="pill">Urdu + English + Roman Urdu</span>
</div>
""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["🔒 Privacy Detection", "✍️ Text Generation", "📚 BERT vs GPT-2"])

# ------------------------- Tab 1: BERT classifier -------------------------
with tab1:
    st.subheader("Does this text contain private information?")
    st.caption("Class 0 = Non-Sensitive  |  Class 1 = Sensitive "
               "(phone, CNIC, email, address, bank account, password, OTP, card, passport, date of birth)")

    EXAMPLES = [
        ("📞 Phone", "میرا فون نمبر 0300-1234567 ہے"),
        ("🪪 CNIC", "My CNIC number is 35202-1234567-1"),
        ("📧 Email", "Mera email ali123@gmail.com hai"),
        ("📚 Study", "I study computer science."),
        ("☀️ موسم", "آج موسم بہت اچھا ہے۔"),
        ("📅 Meeting", "Kal meeting 5 baje hai"),
    ]

    if "msg" not in st.session_state:
        st.session_state.msg = EXAMPLES[1][1]

    def set_msg(text):
        st.session_state.msg = text

    st.write("Try a sample:")
    cols = st.columns(3)
    for i, (label, text) in enumerate(EXAMPLES):
        cols[i % 3].button(label, on_click=set_msg, args=(text,), key=f"ex_{i}")

    st.text_area("Text (Urdu, English or Roman Urdu)", key="msg", height=110)

    if st.button("🔍 Analyze text", type="primary", key="analyze"):
        message = st.session_state.msg.strip()
        if not message:
            st.warning("Please type some text first.")
        else:
            try:
                with st.spinner("Loading BERT model (the first time takes a minute)..."):
                    clf = load_model("text-classification", BERT_REPO)
                result = clf(message, top_k=None, truncation=True, max_length=96)
                if isinstance(result[0], list):
                    result = result[0]

                sensitive_score = sum(r["score"] for r in result if is_sensitive(r["label"]))
                safe_score = 1.0 - sensitive_score

                st.markdown(
                    f'<div class="msgbox" dir="auto">{html.escape(message)}</div>'
                    f'<span class="badge">Language (guess): {detect_language(message)}</span>',
                    unsafe_allow_html=True,
                )

                if sensitive_score >= 0.5:
                    st.markdown(
                        f'<div class="result result-red"><h2>🔒 Sensitive</h2>'
                        f"<p>This text looks like it contains or asks for personal information. "
                        f"Confidence: {sensitive_score:.1%}</p></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="result result-green"><h2>✅ Non-Sensitive</h2>'
                        f"<p>No private information detected. Confidence: {safe_score:.1%}</p></div>",
                        unsafe_allow_html=True,
                    )

                st.progress(float(sensitive_score), text=f"Sensitive (class 1): {sensitive_score:.1%}")
                st.progress(float(safe_score), text=f"Non-Sensitive (class 0): {safe_score:.1%}")
                st.caption("The model can make mistakes, for example when a sentence only says "
                           "\"my CNIC number is as follows\" without the number itself.")
            except Exception as e:
                st.error(
                    "Could not load or run the BERT model. Check that BERT_REPO in app.py "
                    "matches your Hugging Face model name and that the model is public."
                )
                st.caption(f"Technical detail: {e}")

    with st.expander("📊 Model performance (test results)"):
        st.markdown(
            """
**Large test set (31,065 samples)**

| Class | Precision | Recall | F1 | Samples |
|---|---|---|---|---|
| Non-Sensitive | 0.948 | 0.852 | 0.898 | 637 |
| Sensitive | 0.997 | 0.999 | 0.998 | 30,428 |

Accuracy: **99.6%**. Most test samples are *Sensitive*, so accuracy looks very high. The
Non-Sensitive F1 (0.898) is the fairer number.

**Balanced synthetic test set (181 samples):** accuracy **99.5%**.
"""
        )

# ------------------------- Tab 2: GPT-2 generator -------------------------
with tab2:
    st.subheader("Continue my sentence")
    st.caption("Give the beginning of an English sentence (news style works best) and GPT-2 will continue it.")

    PROMPTS = [
        "Oil prices rose sharply on Monday after",
        "The national team won the championship after",
        "Scientists at the university have developed",
        "The president said on Tuesday that",
    ]

    if "prompt" not in st.session_state:
        st.session_state.prompt = PROMPTS[0]

    def set_prompt(text):
        st.session_state.prompt = text

    st.write("Quick prompts:")
    pcols = st.columns(2)
    for i, p in enumerate(PROMPTS):
        pcols[i % 2].button(p, on_click=set_prompt, args=(p,), key=f"pr_{i}")

    st.text_input("Start of the sentence", key="prompt")

    with st.expander("⚙️ Generation settings"):
        n_tokens = st.slider("Number of new tokens", 20, 120, 50)
        temperature = st.slider("Temperature (higher = more creative)", 0.3, 1.5, 0.8)
        top_p = st.slider("Top-p", 0.5, 1.0, 0.92)
        rep_penalty = st.slider("Repetition penalty (higher = less repetition)", 1.0, 1.5, 1.2)

    if st.button("✨ Generate text", type="primary", key="generate"):
        prompt = st.session_state.prompt.strip()
        if not prompt:
            st.warning("Please type the beginning of a sentence.")
        else:
            try:
                with st.spinner("Loading GPT-2 model (the first time takes a minute)..."):
                    gen = load_model("text-generation", GPT_REPO)
                with st.spinner("Generating..."):
                    out = gen(
                        prompt,
                        max_new_tokens=n_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=50,
                        repetition_penalty=rep_penalty,
                        no_repeat_ngram_size=3,
                        pad_token_id=50256,
                    )
                full = out[0]["generated_text"]
                rest = full[len(prompt):] if full.startswith(prompt) else full
                st.markdown(
                    f'<div class="gen"><span class="prompt">{html.escape(prompt)}</span>'
                    f'{html.escape(rest).replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True,
                )
                st.caption("Purple text = your input. The rest was generated token by token. "
                           "Click Generate again to get a different result. "
                           "The output may not be factually correct.")
            except Exception as e:
                st.error(
                    "Could not load or run the GPT-2 model. Check that GPT_REPO in app.py "
                    "matches your Hugging Face model name and that the model is public."
                )
                st.caption(f"Technical detail: {e}")

# --------------------------- Tab 3: comparison ---------------------------
with tab3:
    st.subheader("Encoder-only vs decoder-only")
    st.markdown(
        """
| | **BERT (encoder-only)** | **GPT-2 (decoder-only)** |
|---|---|---|
| Main purpose | Understand text | Generate text |
| Attention | Bidirectional (sees left and right words) | Causal (sees only previous tokens) |
| Output in this app | Sensitive / Non-Sensitive | A continuation of your sentence |
| Typical tasks | Classification, NER, question answering | Text completion, writing, chat |
| Model used | Fine-tuned `bert-base-multilingual-cased` | `gpt2` fine-tuned on AG News (English news) |
"""
    )
    st.info(
        "**Key idea:** the encoder reads the whole input at once to build understanding, "
        "while the decoder predicts the next token again and again to generate new text."
    )

st.markdown('<div class="footer">Built with Hugging Face Transformers and Streamlit</div>',
            unsafe_allow_html=True)
