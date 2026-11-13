import html
import re
import time

import streamlit as st
from transformers import pipeline

# =====================================================================
#  SETTINGS - replace YOUR_GITHUB_USERNAME (Hugging Face names are set)
# =====================================================================
BERT_REPO = "Maryam033/bert-privacy-detector"
GPT_REPO = "Maryam033/gpt2-text-generation"
GPT_FALLBACK = "openai-community/gpt2"   # used only if GPT_REPO cannot be loaded
GITHUB_URL = "https://github.com/YOUR_GITHUB_USERNAME/bert-gpt2-privacy-app"
# =====================================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

.stApp, .stApp p, .stApp label, .stApp li, .stApp button, .stApp textarea, .stApp input,
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {font-family: 'Poppins', 'Segoe UI', system-ui, sans-serif;}
.stApp {
  background-image:
    radial-gradient(900px 420px at 8% -8%, rgba(99,102,241,.20), transparent 70%),
    radial-gradient(800px 380px at 100% 0%, rgba(236,72,153,.15), transparent 70%),
    radial-gradient(700px 400px at 50% 110%, rgba(14,165,233,.12), transparent 70%);
}
.block-container {padding-top: 1.6rem; max-width: 900px;}
footer {visibility: hidden;}

/* ---------- hero ---------- */
.hero {
  position: relative; overflow: hidden; color: #fff; border-radius: 26px; padding: 2.3rem 2rem 1.8rem 2rem;
  background: linear-gradient(120deg, #4338ca, #7c3aed, #db2777, #4338ca); background-size: 300% 300%;
  animation: shift 12s ease infinite; box-shadow: 0 18px 50px rgba(99,102,241,.35);
}
@keyframes shift {0% {background-position: 0% 50%} 50% {background-position: 100% 50%} 100% {background-position: 0% 50%}}
.hero:before, .hero:after {content: ""; position: absolute; border-radius: 50%; background: rgba(255,255,255,.10);}
.hero:before {width: 260px; height: 260px; right: -70px; top: -90px;}
.hero:after {width: 160px; height: 160px; right: 120px; bottom: -90px;}
.hero-badge {
  display: inline-block; padding: .25rem .9rem; border-radius: 999px; font-size: .78rem; font-weight: 600;
  background: rgba(255,255,255,.20); border: 1px solid rgba(255,255,255,.35); letter-spacing: .04em;
}
.hero h1 {color: #fff; font-size: 2.6rem; font-weight: 800; line-height: 1.15; margin: .7rem 0 .4rem 0; padding: 0;}
.hero h1 span {opacity: .85;}
.hero p {color: rgba(255,255,255,.92); font-size: 1.05rem; max-width: 560px; margin: 0 0 1.3rem 0;}
.hero-stats {display: flex; gap: 1.8rem; flex-wrap: wrap; position: relative; z-index: 1;}
.hero-stats div {display: flex; flex-direction: column;}
.hero-stats b {font-size: 1.6rem; font-weight: 700; line-height: 1.1;}
.hero-stats small {font-size: .78rem; opacity: .85; text-transform: uppercase; letter-spacing: .06em;}

/* ---------- feature cards ---------- */
.features {display: grid; grid-template-columns: repeat(3, 1fr); gap: .9rem; margin: 1.2rem 0 1rem 0;}
.feature {
  border-radius: 18px; padding: 1rem 1.1rem; background: rgba(128,128,128,.08);
  border: 1px solid rgba(128,128,128,.22); transition: transform .2s ease, box-shadow .2s ease;
}
.feature:hover {transform: translateY(-4px); box-shadow: 0 10px 26px rgba(99,102,241,.22);}
.feature .ico {font-size: 1.7rem;}
.feature h4 {margin: .3rem 0 .15rem 0; font-size: 1rem; font-weight: 700; padding: 0;}
.feature p {margin: 0; font-size: .85rem; opacity: .75;}
@media (max-width: 700px) {.features {grid-template-columns: 1fr;} .hero h1 {font-size: 2rem;}}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] {gap: 6px; background: rgba(128,128,128,.12); padding: 6px; border-radius: 16px;}
.stTabs [data-baseweb="tab"] {border-radius: 12px; padding: .45rem 1rem; font-weight: 600; height: auto;}
.stTabs [aria-selected="true"] {background: linear-gradient(120deg, #6366f1, #a855f7); color: #fff !important;}
.stTabs [aria-selected="true"] p {color: #fff !important;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display: none;}

/* ---------- inputs and buttons ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 20px;}
.stTextArea textarea, .stTextInput input {border-radius: 14px; font-size: 1rem;}
button[kind="primary"], button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(120deg, #6366f1, #a855f7, #ec4899); background-size: 200% 200%; color: #fff; border: 0;
  border-radius: 14px; padding: .55rem 1.4rem; font-weight: 700; transition: transform .15s ease, box-shadow .15s ease;
  box-shadow: 0 8px 22px rgba(124,58,237,.35);
}
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
  transform: translateY(-2px); box-shadow: 0 12px 28px rgba(124,58,237,.5); color: #fff; border: 0;
}
button[kind="secondary"], button[data-testid="stBaseButton-secondary"] {
  border-radius: 999px; font-weight: 500; border: 1px solid rgba(99,102,241,.45); background: rgba(99,102,241,.08);
}
button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover {
  border-color: #6366f1; background: rgba(99,102,241,.18); color: inherit;
}
.section-title {font-size: 1.15rem; font-weight: 700; margin: .1rem 0 .1rem 0;}
.muted {opacity: .7; font-size: .9rem;}

/* ---------- result card ---------- */
.quote {
  border-left: 4px solid #6366f1; background: rgba(99,102,241,.08); border-radius: 0 12px 12px 0;
  padding: .7rem 1rem; margin: 1rem 0 .8rem 0; font-size: 1.02rem;
}
.res {display: flex; gap: 1.4rem; align-items: center; flex-wrap: wrap; border-radius: 22px; padding: 1.3rem 1.5rem;
  animation: pop .5s ease both;}
@keyframes pop {from {opacity: 0; transform: translateY(12px) scale(.98);} to {opacity: 1; transform: none;}}
.res-danger {background: rgba(239,68,68,.10); border: 1px solid rgba(239,68,68,.38);}
.res-safe {background: rgba(34,197,94,.10); border: 1px solid rgba(34,197,94,.38);}
.res-gauge {flex: 0 0 auto; display: flex; flex-direction: column; align-items: center;}
.gauge-cap {font-size: .72rem; font-weight: 600; letter-spacing: .14em; opacity: .75; margin-top: -.2rem;}
.res-body {flex: 1 1 260px;}
.res-body h3 {margin: .5rem 0 .2rem 0; font-size: 1.35rem; font-weight: 700; padding: 0;}
.res-body p {margin: 0 0 .6rem 0; opacity: .8; font-size: .92rem;}
.chip-level {display: inline-block; color: #fff; padding: .15rem .75rem; border-radius: 999px; font-size: .72rem;
  font-weight: 700; letter-spacing: .06em;}
.badge {display: inline-block; padding: .12rem .7rem; border-radius: 999px; font-size: .8rem; margin-bottom: .6rem;
  background: rgba(99,102,241,.15); color: #6366f1; border: 1px solid rgba(99,102,241,.35);}
.bar-row {display: flex; justify-content: space-between; font-size: .82rem; margin-top: .35rem; opacity: .85;}
.bar {height: 9px; border-radius: 999px; background: rgba(128,128,128,.22); overflow: hidden;}
.bar-fill {height: 100%; border-radius: 999px; animation: grow 1s ease-out both;}
@keyframes grow {from {width: 0;}}
.ring {animation: ring-fill 1.1s ease-out both;}
@keyframes ring-fill {from {stroke-dashoffset: var(--full);}}

/* ---------- generated text ---------- */
.paper {
  white-space: pre-wrap; line-height: 1.8; font-size: 1.05rem; border-radius: 20px; padding: 1.3rem 1.5rem;
  background: rgba(99,102,241,.07); border: 1px solid rgba(99,102,241,.28); border-left: 6px solid #8b5cf6;
  animation: pop .5s ease both;
}
.p-prompt {font-weight: 700; color: #7c3aed;}
@supports ((-webkit-background-clip: text) or (background-clip: text)) {
  .p-prompt {
    background: linear-gradient(90deg, #6366f1, #ec4899); -webkit-background-clip: text;
    background-clip: text; -webkit-text-fill-color: transparent;
  }
}
.stats {display: flex; gap: .5rem; flex-wrap: wrap; margin: .8rem 0 .3rem 0;}
.stat {padding: .2rem .8rem; border-radius: 999px; font-size: .8rem; background: rgba(128,128,128,.14);
  border: 1px solid rgba(128,128,128,.25);}

/* ---------- comparison ---------- */
.cmp {display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: .8rem 0;}
@media (max-width: 700px) {.cmp {grid-template-columns: 1fr;}}
.cmp-card {border-radius: 20px; padding: 1.2rem 1.3rem; background: rgba(128,128,128,.08);
  border: 1px solid rgba(128,128,128,.25);}
.cmp-card h4 {margin: 0 0 .2rem 0; font-size: 1.2rem; font-weight: 700; padding: 0;}
.tag {display: inline-block; padding: .1rem .7rem; border-radius: 999px; font-size: .75rem; font-weight: 600; color: #fff;
  background: linear-gradient(120deg, #6366f1, #a855f7); margin-bottom: .8rem;}
.tag.pink {background: linear-gradient(120deg, #ec4899, #f97316);}
.attn {border-spacing: 4px; margin: .2rem 0 .3rem 0;}
.attn td {width: 24px; height: 24px; border-radius: 6px;}
.attn td.on {background: linear-gradient(135deg, #6366f1, #a855f7);}
.attn td.on.pink {background: linear-gradient(135deg, #ec4899, #f97316);}
.attn td.off {background: rgba(128,128,128,.16);}
.cap {font-size: .82rem; opacity: .75; margin: 0 0 .6rem 0;}
.cmp-card ul {margin: 0; padding-left: 1.1rem; font-size: .9rem;}
.flow {display: flex; align-items: center; justify-content: center; gap: .5rem; flex-wrap: wrap; margin: 1rem 0;}
.flow span {padding: .45rem 1rem; border-radius: 12px; font-weight: 600; font-size: .9rem;
  background: rgba(99,102,241,.12); border: 1px solid rgba(99,102,241,.35);}
.flow i {font-style: normal; opacity: .6;}

/* ---------- sidebar and footer ---------- */
.side-card {border-radius: 16px; padding: 1rem; color: #fff; margin-bottom: .8rem;
  background: linear-gradient(135deg, #4338ca, #a855f7);}
.side-card b {font-size: 1.05rem;}
.side-card small {display: block; opacity: .85; margin-top: .2rem;}
.foot {text-align: center; opacity: .6; font-size: .82rem; margin: 2rem 0 .5rem 0;}
</style>
"""

URDU_RE = re.compile(r"[\u0600-\u06FF]")
ROMAN_URDU_WORDS = {
    "hai", "hain", "mera", "meri", "mujhe", "kal", "aaj", "nahi", "kya", "aap",
    "mein", "baje", "raha", "rahi", "hoon", "karo", "kar", "ka", "ki", "shukriya",
}


def esc(text):
    return html.escape(str(text))


def detect_language(text):
    if URDU_RE.search(text):
        return "Urdu"
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    if len(words & ROMAN_URDU_WORDS) >= 2:
        return "Roman Urdu (guess)"
    return "English"


def label_is_sensitive(label):
    # works with "Sensitive" / "Non-Sensitive" and with LABEL_0 / LABEL_1
    return str(label).strip().lower() in {"sensitive", "label_1", "1"}


def hero_html():
    return "".join([
        '<div class="hero">',
        '<span class="hero-badge">✨ TRANSFORMER STUDIO</span>',
        '<h1>BERT <span>&amp;</span> GPT-2 Studio</h1>',
        '<p>Spot private information in Urdu, English and Roman Urdu, then watch GPT-2 write news-style text.</p>',
        '<div class="hero-stats">',
        '<div><b>2</b><small>Models</small></div>',
        '<div><b>3</b><small>Languages</small></div>',
        '<div><b>150K+</b><small>Training texts</small></div>',
        '</div></div>',
    ])


def features_html():
    cards = [
        ("🧠", "BERT understands", "Reads a whole sentence at once and decides if it contains private data."),
        ("✍️", "GPT-2 writes", "Predicts the next word again and again to continue your sentence."),
        ("🌍", "Three languages", "Urdu script, English and Roman Urdu, all in one model."),
    ]
    return '<div class="features">' + "".join(
        f'<div class="feature"><div class="ico">{i}</div><h4>{t}</h4><p>{d}</p></div>' for i, t, d in cards
    ) + "</div>"


def gauge_svg(score, color, label):
    circ = 2 * 3.14159 * 52
    offset = circ * (1 - score)
    return "".join([
        '<svg viewBox="0 0 120 120" width="150" height="150" role="img">',
        '<circle cx="60" cy="60" r="52" fill="none" stroke="rgba(128,128,128,.25)" stroke-width="10"/>',
        f'<circle class="ring" cx="60" cy="60" r="52" fill="none" stroke="{color}" stroke-width="10" ',
        f'stroke-linecap="round" stroke-dasharray="{circ:.1f}" stroke-dashoffset="{offset:.1f}" ',
        f'transform="rotate(-90 60 60)" style="--full:{circ:.1f}"/>',
        f'<text x="60" y="68" text-anchor="middle" font-size="26" font-weight="700" fill="currentColor">{score:.0%}</text>',
        '</svg>',
    ])


def bar_html(label, value, color):
    return (f'<div class="bar-row"><span>{label}</span><span>{value:.1%}</span></div>'
            f'<div class="bar"><div class="bar-fill" style="width:{value * 100:.1f}%;background:{color}"></div></div>')


def result_html(message, sensitive):
    safe = 1.0 - sensitive
    if sensitive >= 0.5:
        tone, color = "danger", "#ef4444"
        level = "HIGH CONFIDENCE" if sensitive >= 0.8 else "LIKELY SENSITIVE"
        title = "Sensitive information detected"
        advice = "This text looks like it contains or asks for personal data. Think twice before sharing it."
        gscore, glabel = sensitive, "SENSITIVE"
    else:
        tone, color = "safe", "#22c55e"
        level = "LOOKS SAFE" if safe >= 0.8 else "PROBABLY SAFE"
        title = "No sensitive information detected"
        advice = "No private details were found. The model can still make mistakes, so stay careful."
        gscore, glabel = safe, "SAFE"
    return "".join([
        f'<div class="quote" dir="auto">{esc(message)}</div>',
        f'<div class="res res-{tone}">',
        f'<div class="res-gauge">{gauge_svg(gscore, color, glabel)}<div class="gauge-cap">{glabel}</div></div>',
        '<div class="res-body">',
        f'<span class="chip-level" style="background:{color}">{level}</span>',
        f'<h3>{title}</h3><p>{advice}</p>',
        f'<span class="badge">🌐 {esc(detect_language(message))}</span>',
        bar_html("Sensitive (class 1)", sensitive, "#ef4444"),
        bar_html("Non-Sensitive (class 0)", safe, "#22c55e"),
        '</div></div>',
    ])


def generated_html(prompt, continuation):
    return f'<div class="paper"><span class="p-prompt">{esc(prompt)}</span>{esc(continuation)}</div>'


def attention_grid(kind, n=6):
    rows = ""
    for i in range(n):
        cells = ""
        for j in range(n):
            if kind == "bert":
                cells += '<td class="on"></td>'
            else:
                cells += '<td class="on pink"></td>' if j <= i else '<td class="off"></td>'
        rows += f"<tr>{cells}</tr>"
    return f'<table class="attn">{rows}</table>'


def compare_html():
    bert = "".join([
        '<div class="cmp-card"><h4>🧠 BERT</h4><span class="tag">Encoder-only</span>',
        attention_grid("bert"),
        '<p class="cap">Every word looks at every other word, left and right.</p>',
        '<ul><li>Understands and classifies text</li><li>Used for: Sensitive vs Non-Sensitive</li>',
        '<li>Output: a label with a confidence</li></ul></div>',
    ])
    gpt = "".join([
        '<div class="cmp-card"><h4>✍️ GPT-2</h4><span class="tag pink">Decoder-only</span>',
        attention_grid("gpt"),
        '<p class="cap">Each word looks only at the words before it.</p>',
        '<ul><li>Generates and continues text</li><li>Used for: news-style text</li>',
        '<li>Output: new text, token by token</li></ul></div>',
    ])
    return f'<div class="cmp">{bert}{gpt}</div>'


def flow_html():
    steps = ["Google Colab", "Hugging Face", "GitHub", "Streamlit"]
    return '<div class="flow">' + "<i>→</i>".join(f"<span>{s}</span>" for s in steps) + "</div>"


# One shared cache with max_entries=1: only ONE model stays in memory at a time.
@st.cache_resource(show_spinner=False, max_entries=1)
def _load(task, repo):
    return pipeline(task, model=repo)


def load_classifier():
    return _load("text-classification", BERT_REPO)


def load_generator():
    """Load the fine-tuned GPT-2; if that repo cannot be loaded, use the original GPT-2."""
    try:
        return _load("text-generation", GPT_REPO), GPT_REPO
    except Exception:
        st.warning(f"Could not load `{GPT_REPO}`, so the original `{GPT_FALLBACK}` is used instead.")
        return _load("text-generation", GPT_FALLBACK), GPT_FALLBACK


def classify(message):
    predictions = load_classifier()(message, truncation=True, max_length=96, top_k=None)
    if isinstance(predictions, dict):
        predictions = [predictions]
    if predictions and isinstance(predictions[0], list):
        predictions = predictions[0]
    score = sum(float(p["score"]) for p in predictions if label_is_sensitive(p["label"]))
    return max(0.0, min(1.0, score))


def repo_exists(repo):
    try:
        from huggingface_hub import HfApi

        HfApi().model_info(repo)
        return True
    except Exception:
        return False


# ============================== UI START ==============================
st.set_page_config(page_title="BERT & GPT-2 Studio", page_icon="🔒", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------- Sidebar ---------------------------------
with st.sidebar:
    st.markdown('<div class="side-card"><b>🔒 BERT &amp; GPT-2 Studio</b>'
                '<small>A class project comparing an encoder-only and a decoder-only Transformer.</small></div>',
                unsafe_allow_html=True)
    st.markdown("**Models and code**")
    links = [("🤗 BERT Privacy Detector", f"https://huggingface.co/{BERT_REPO}"),
             ("🤗 GPT-2 Text Generation", f"https://huggingface.co/{GPT_REPO}"),
             ("💻 GitHub repository", GITHUB_URL)]
    for text, url in links:
        if hasattr(st, "link_button"):
            st.link_button(text, url)
        else:
            st.markdown(f"- [{text}]({url})")
    st.markdown("---")
    if st.button("🩺 Check models"):
        for name, repo in [("BERT", BERT_REPO), ("GPT-2", GPT_REPO)]:
            if repo_exists(repo):
                st.success(f"{name}: found on Hugging Face")
            else:
                st.error(f"{name}: `{repo}` not found. Check the name and that it is Public.")
    st.warning("Use **fake sample data** only. Never enter your real CNIC, phone number, password, "
               "card number or OTP. Your text is not saved by this app.")

# --------------------------------- Header ---------------------------------
st.markdown(hero_html(), unsafe_allow_html=True)
st.markdown(features_html(), unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔒 Privacy Detector", "✍️ Text Generator", "📚 BERT vs GPT-2"])

# ------------------------------- Privacy tab -------------------------------
with tab1:
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

    def use_example(text):
        st.session_state.msg = text
        st.session_state.pop("bert_result", None)

    with st.container(border=True):
        st.markdown('<div class="section-title">Is there private information in this text?</div>'
                    '<div class="muted">Type Urdu, English or Roman Urdu, or tap a sample.</div>',
                    unsafe_allow_html=True)
        cols = st.columns(3)
        for i, (label, text) in enumerate(EXAMPLES):
            cols[i % 3].button(label, on_click=use_example, args=(text,), key=f"ex_{i}")
        st.text_area("Your text", key="msg", height=110, max_chars=1000,
                     placeholder="Type or paste a sample message here...", label_visibility="collapsed")
        analyze = st.button("🔍 Analyze text", type="primary", key="analyze")

    if analyze:
        text = st.session_state.msg.strip()
        if not text:
            st.warning("Please type some text first.")
        else:
            try:
                with st.spinner("Loading BERT (the first time takes a minute)..."):
                    score = classify(text)
                st.session_state.bert_result = {"message": text, "score": score}
            except Exception as error:
                st.session_state.pop("bert_result", None)
                st.error("Could not load or run the BERT model. Check BERT_REPO in app.py, make sure the "
                         "model is Public, and use 🩺 Check models in the sidebar.")
                st.caption(f"Technical detail: {error}")

    if "bert_result" in st.session_state:
        res = st.session_state.bert_result
        st.markdown(result_html(res["message"], res["score"]), unsafe_allow_html=True)
        st.caption("Class 0 = Non-Sensitive, Class 1 = Sensitive. This is a student project and can make mistakes, "
                   "for example when a sentence only says \"my CNIC number is as follows\" without the number.")

# --------------------------------- GPT-2 tab ---------------------------------
with tab2:
    PROMPTS = [
        "Oil prices rose sharply on Monday after",
        "The national team won the championship after",
        "Scientists at the university have developed",
        "The president said on Tuesday that",
    ]
    if "prompt" not in st.session_state:
        st.session_state.prompt = PROMPTS[0]

    def use_prompt(text):
        st.session_state.prompt = text
        st.session_state.pop("gpt_result", None)

    with st.container(border=True):
        st.markdown('<div class="section-title">Let GPT-2 continue your sentence</div>'
                    '<div class="muted">News-style English prompts work best.</div>', unsafe_allow_html=True)
        pcols = st.columns(2)
        for i, p in enumerate(PROMPTS):
            pcols[i % 2].button(p, on_click=use_prompt, args=(p,), key=f"pr_{i}")
        st.text_area("Starting prompt", key="prompt", height=90, max_chars=300,
                     label_visibility="collapsed", placeholder="Type the beginning of a sentence...")
        with st.expander("⚙️ Creativity settings"):
            c1, c2 = st.columns(2)
            n_tokens = c1.slider("New tokens", 20, 120, 50)
            temperature = c1.slider("Creativity (temperature)", 0.3, 1.5, 0.8, 0.1)
            top_p = c2.slider("Top-p", 0.5, 1.0, 0.92, 0.01)
            rep_penalty = c2.slider("Repetition penalty", 1.0, 1.5, 1.2, 0.05)
        generate = st.button("✨ Generate text", type="primary", key="generate")

    if generate:
        prompt = st.session_state.prompt.strip()
        if not prompt:
            st.warning("Please type the beginning of a sentence.")
        else:
            try:
                with st.spinner("Loading GPT-2 (the first time takes a minute)..."):
                    generator, used_repo = load_generator()
                started = time.perf_counter()
                with st.spinner("Writing..."):
                    out = generator(prompt, max_new_tokens=n_tokens, do_sample=True, temperature=temperature,
                                    top_p=top_p, top_k=50, repetition_penalty=rep_penalty,
                                    no_repeat_ngram_size=3, pad_token_id=50256)
                full = out[0]["generated_text"]
                rest = full[len(prompt):] if full.startswith(prompt) else full
                st.session_state.gpt_result = {"prompt": prompt, "rest": rest, "full": full,
                                               "secs": time.perf_counter() - started, "repo": used_repo}
            except Exception as error:
                st.session_state.pop("gpt_result", None)
                st.error("Could not load or run GPT-2. Check GPT_REPO in app.py, make sure the model is Public, "
                         "and use 🩺 Check models in the sidebar.")
                st.caption(f"Technical detail: {error}")

    if "gpt_result" in st.session_state:
        g = st.session_state.gpt_result
        st.markdown(generated_html(g["prompt"], g["rest"]), unsafe_allow_html=True)
        st.markdown(
            f'<div class="stats"><span class="stat">📝 {len(g["full"].split())} words</span>'
            f'<span class="stat">⏱ {g["secs"]:.1f} s</span><span class="stat">🤖 {esc(g["repo"])}</span></div>',
            unsafe_allow_html=True)
        with st.expander("📋 Copy the text"):
            st.code(g["full"], language=None)
        st.caption("Purple text is your input. GPT-2 writes likely text, not verified news, so check facts "
                   "before sharing. Press Generate again for a different result.")

# --------------------------------- Compare tab ---------------------------------
with tab3:
    st.markdown('<div class="section-title">Encoder-only vs decoder-only</div>'
                '<div class="muted">Each square shows which words a word can look at.</div>',
                unsafe_allow_html=True)
    st.markdown(compare_html(), unsafe_allow_html=True)
    st.info("**Key idea:** BERT reads the whole input at once to understand it. GPT-2 predicts the next "
            "token again and again to generate new text.")
    st.markdown('<div class="section-title">How this project fits together</div>', unsafe_allow_html=True)
    st.markdown(flow_html(), unsafe_allow_html=True)
    st.caption("Colab trains the models, Hugging Face stores them, GitHub stores the code, and Streamlit runs "
               "this app.")

st.markdown('<div class="foot">Built with Streamlit and Hugging Face Transformers · Educational demo only</div>',
            unsafe_allow_html=True)
