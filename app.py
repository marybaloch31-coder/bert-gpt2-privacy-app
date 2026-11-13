import html
import re

import streamlit as st
from transformers import pipeline

# =====================================================================
#  SETTINGS - replace YOUR_GITHUB_USERNAME (Hugging Face names are set)
# =====================================================================
BERT_REPO = "Maryam033/bert-privacy-detector"
GPT_REPO = "Maryam033/gpt2-text-generation"
GITHUB_URL = "https://github.com/YOUR_GITHUB_USERNAME/bert-gpt2-privacy-app"
# =====================================================================

st.set_page_config(page_title="BERT & GPT-2 Studio", page_icon="🔒", layout="centered")

st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 860px;}
.hero {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 50%, #a855f7 100%);
    color: #ffffff; padding: 1.8rem 1.6rem; border-radius: 18px; margin-bottom: 1rem;
    box-shadow: 0 10px 30px rgba(99, 102, 241, 0.25);
}
.hero h1 {color: #ffffff; margin: 0 0 .3rem 0; font-size: 1.9rem; line-height: 1.25;}
.hero p {color: rgba(255,255,255,.92); margin: 0; font-size: 1rem;}
.result {border-radius: 14px; padding: 1rem 1.3rem; margin: 1rem 0 .6rem 0;}
.result h3 {margin: 0 0 .2rem 0;}
.result p {margin: 0; opacity: .85;}
.result.sensitive {background: rgba(239,68,68,.12); border-left: 6px solid #ef4444;}
.result.safe {background: rgba(34,197,94,.12); border-left: 6px solid #22c55e;}
.generated {
    background: rgba(99,102,241,.08); border: 1px solid rgba(99,102,241,.3);
    border-radius: 14px; padding: 1.1rem 1.3rem; line-height: 1.75; font-size: 1.05rem;
    white-space: pre-wrap;
}
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
        return "Roman Urdu 🇵🇰 (guess)"
    return "English 🇬🇧"


def label_is_sensitive(label):
    # works with "Sensitive" / "Non-Sensitive" and with LABEL_0 / LABEL_1
    return str(label).strip().lower() in {"sensitive", "label_1", "1"}


# One shared cache with max_entries=1: only ONE model stays in memory at a time.
# (Two separate cached loaders would keep BERT and GPT-2 in memory together.)
@st.cache_resource(show_spinner=False, max_entries=1)
def _load(task, repo):
    return pipeline(task, model=repo)


def load_classifier(repo):
    return _load("text-classification", repo)


def load_generator(repo):
    return _load("text-generation", repo)


def show_model_error(name, error):
    st.error(
        f"Could not load or run the {name}. Check that the model name in app.py is correct, "
        "that the model is Public on Hugging Face, and use the 🩺 Check models button in the sidebar."
    )
    st.caption(f"Technical detail: {error}")


def repo_exists(repo):
    try:
        from huggingface_hub import HfApi

        HfApi().model_info(repo)
        return True
    except Exception:
        return False


# --------------------------------- Sidebar ---------------------------------
with st.sidebar:
    st.markdown("## 📌 About")
    st.write("A class project demonstrating two Transformer models:")
    st.markdown("- **BERT:** detects privacy-sensitive text.\n- **GPT-2:** continues a text prompt.")
    st.markdown("### Model links")
    st.markdown(f"- [BERT Privacy Detector](https://huggingface.co/{BERT_REPO})")
    st.markdown(f"- [GPT-2 Text Generation](https://huggingface.co/{GPT_REPO})")
    st.markdown("### Source code")
    st.markdown(f"- [GitHub repository]({GITHUB_URL})")
    if st.button("🩺 Check models"):
        for model_name, repo in [("BERT", BERT_REPO), ("GPT-2", GPT_REPO)]:
            if repo_exists(repo):
                st.success(f"{model_name}: found on Hugging Face")
            else:
                st.error(f"{model_name}: `{repo}` not found. Check the name and that it is Public.")
    st.warning(
        "Use fake sample data only. Do not enter your real CNIC, phone number, "
        "password, card number or OTP."
    )

# ------------------------------- Main page -------------------------------
st.markdown(
    """
<div class="hero">
  <h1>🔒 BERT &amp; GPT-2 Studio</h1>
  <p>Detect privacy-sensitive messages and generate text with Transformer models.</p>
</div>
""",
    unsafe_allow_html=True,
)
st.write("Choose a tool below to get started.")

tab1, tab2, tab3 = st.tabs(
    ["🔒 Privacy Detection", "✍️ GPT-2 Text Generation", "📚 BERT vs GPT-2"]
)

# -------------------------- BERT privacy checker --------------------------
with tab1:
    st.subheader("Privacy-sensitive text detector")
    st.write(
        "Enter a short message in Urdu, English or Roman Urdu. "
        "The model predicts whether it contains sensitive personal information."
    )
    st.caption("Class 0 = Non-Sensitive | Class 1 = Sensitive. Use fictional examples only.")

    EXAMPLES = {
        "Phone example": "My phone number is 0300-1234567",
        "CNIC example": "My CNIC number is 35202-1234567-1",
        "Email example": "Mera email ali123@gmail.com hai",
        "Normal example": "I study computer science.",
        "Urdu example": "آج موسم بہت اچھا ہے۔",
    }

    if "privacy_message" not in st.session_state:
        st.session_state.privacy_message = ""

    def fill_example():
        # runs BEFORE the page is redrawn, so the text box shows the chosen example
        choice = st.session_state.example_choice
        if choice in EXAMPLES:
            st.session_state.privacy_message = EXAMPLES[choice]

    st.selectbox(
        "Try an example",
        ["Choose an example"] + list(EXAMPLES),
        key="example_choice",
        on_change=fill_example,
    )
    message = st.text_area(
        "Your message",
        height=120,
        max_chars=1000,
        placeholder="Type or paste a sample message here...",
        key="privacy_message",
    )

    if st.button("🔍 Analyze message", type="primary", key="analyze_button"):
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            try:
                with st.spinner("Loading the BERT model. The first load may take a minute..."):
                    classifier = load_classifier(BERT_REPO)
                predictions = classifier(
                    message.strip(), truncation=True, max_length=96, top_k=None
                )
                if isinstance(predictions, dict):
                    predictions = [predictions]
                if predictions and isinstance(predictions[0], list):
                    predictions = predictions[0]

                sensitive_score = sum(
                    float(item["score"])
                    for item in predictions
                    if label_is_sensitive(item["label"])
                )
                sensitive_score = max(0.0, min(1.0, sensitive_score))
                safe_score = 1.0 - sensitive_score

                st.markdown(f"**Detected language:** {detect_language(message)}")
                st.markdown("**Text checked:**")
                st.code(message.strip(), language=None)

                if sensitive_score >= 0.5:
                    st.markdown(
                        f'<div class="result sensitive"><h3>🔒 Sensitive information detected</h3>'
                        f"<p>Estimated confidence: {sensitive_score:.1%}</p></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="result safe"><h3>✅ No sensitive information detected</h3>'
                        f"<p>Estimated confidence: {safe_score:.1%}</p></div>",
                        unsafe_allow_html=True,
                    )
                st.progress(sensitive_score, text=f"Sensitive: {sensitive_score:.1%}")
                st.progress(safe_score, text=f"Non-sensitive: {safe_score:.1%}")
                st.caption(
                    "This is a student project and may make mistakes. "
                    "Do not rely on it to protect real personal information."
                )
            except Exception as error:
                show_model_error("BERT privacy detector", error)

# ------------------------------ GPT-2 generator ------------------------------
with tab2:
    st.subheader("GPT-2 text generation")
    st.write("Enter the beginning of an English news-style sentence and GPT-2 will continue it.")
    prompt = st.text_area(
        "Starting prompt",
        value="Scientists have announced a new discovery that",
        height=100,
        max_chars=300,
        key="generation_prompt",
    )
    col1, col2 = st.columns(2)
    with col1:
        max_tokens = st.slider("New tokens", min_value=20, max_value=120, value=50)
        temperature = st.slider("Creativity", min_value=0.3, max_value=1.5, value=0.8, step=0.1)
    with col2:
        top_p = st.slider("Top-p", min_value=0.5, max_value=1.0, value=0.92, step=0.01)
        repetition_penalty = st.slider(
            "Repetition penalty", min_value=1.0, max_value=1.5, value=1.2, step=0.05
        )

    if st.button("✨ Generate text", type="primary", key="generate_button"):
        if not prompt.strip():
            st.warning("Please enter a starting sentence.")
        else:
            try:
                with st.spinner("Loading GPT-2. The first load may take a minute..."):
                    generator = load_generator(GPT_REPO)
                with st.spinner("Generating text..."):
                    output = generator(
                        prompt.strip(),
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=50,
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=3,
                        pad_token_id=generator.tokenizer.eos_token_id,
                    )
                generated_text = output[0]["generated_text"]
                st.markdown("### Generated result")
                st.markdown(
                    f'<div class="generated">{html.escape(generated_text)}</div>',
                    unsafe_allow_html=True,
                )
                st.caption("GPT-2 generates likely text, not verified news. Check facts before sharing.")
            except Exception as error:
                show_model_error("GPT-2 text generator", error)

# ------------------------------ Model comparison ------------------------------
with tab3:
    st.subheader("BERT (encoder) vs GPT-2 (decoder)")
    st.markdown(
        """
| Feature | BERT | GPT-2 |
|---|---|---|
| Architecture | Encoder-only | Decoder-only |
| Main purpose | Understand and classify text | Generate/continue text |
| Attention | Bidirectional | Causal |
| This app's task | Sensitive / non-sensitive classification | News-style text continuation |
"""
    )
    st.info(
        "BERT reads context from both directions to understand text. "
        "GPT-2 predicts the next token repeatedly to generate a continuation."
    )

st.divider()
st.caption("Built with Streamlit and Hugging Face Transformers. Educational demo only.")
