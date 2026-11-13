from pathlib import Path
app_code = r'''import html
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
    st.markdown(f"- [GitHub repository]({GITHUB_URL})")
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
    st.caption(
        "Class 0 = Non-Sensitive | Class 1 = Sensitive. "
        "Use fictional examples only."
    )

    examples = {
        "Phone example": "My phone number is 0300-1234567",
        "CNIC example": "My CNIC number is 35202-1234567-1",
        "Email example": "Mera email ali123@gmail.com hai",
        "Normal example": "I study computer science.",
        "Urdu example": "آج موسم بہت اچھا ہے۔",
    }
    chosen = st.selectbox("Try an example", ["Choose an example"] + list(examples))
    default_text = examples[chosen] if chosen in examples else ""
    message = st.text_area(
        "Your message",
        value=default_text,
        height=120,
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
                    message.strip(),
                    truncation=True,
                    max_length=96,
                    top_k=None,
                )
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
                st.code(message.strip())

                if sensitive_score >= 0.5:
                    st.markdown(
                        f'<div class="result sensitive"><h3>🔒 Sensitive information detected</h3>'
                        f'<p>Estimated confidence: {sensitive_score:.1%}</p></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="result safe"><h3>✅ No sensitive information detected</h3>'
                        f'<p>Estimated confidence: {safe_score:.1%}</p></div>',
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
'''
out=Path("/mnt/data/app.py")
out.write_text(app_code, encoding="utf-8")
import py_compile
py_compile.compile(str(out), doraise=True)
print("Saved", out, "bytes:", out.stat().st_size, "syntax: OK")
