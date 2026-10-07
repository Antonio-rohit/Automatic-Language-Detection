import re
from html import escape

import joblib
import numpy as np
import streamlit as st
from st_keyup import st_keyup

st.set_page_config(page_title="AI Language Detector", page_icon="🌍", layout="centered")

# ============================================================
# STYLING
# ============================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Animated gradient background */
.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(124, 92, 255, 0.22), transparent 40%),
        radial-gradient(circle at 85% 20%, rgba(0, 212, 255, 0.16), transparent 40%),
        radial-gradient(circle at 50% 100%, rgba(255, 92, 168, 0.12), transparent 45%),
        #0b0f1a;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { max-width: 760px; padding-top: 3rem; padding-bottom: 3rem; }

/* Hero */
.hero { text-align: center; margin-bottom: 2rem; }
.hero .globe {
    font-size: 3.2rem;
    display: inline-block;
    animation: float 3.5s ease-in-out infinite;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0.2rem 0 0.4rem 0;
    background: linear-gradient(90deg, #a78bfa, #60a5fa 50%, #34d399);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p { color: #8b95b3; font-size: 1.02rem; margin: 0; }

/* Input label + iframe wrapper (st_keyup lives in an iframe) */
.stApp label p { color: #aeb7d4 !important; font-weight: 600; font-size: 0.9rem; letter-spacing: 0.02em; }
iframe[title="st_keyup.st_keyup"] {
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.10);
    background: rgba(255, 255, 255, 0.04);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
iframe[title="st_keyup.st_keyup"]:hover {
    border-color: rgba(124, 92, 255, 0.55);
    box-shadow: 0 8px 30px rgba(124, 92, 255, 0.20);
}

/* Result card */
.result-card {
    margin-top: 1.5rem;
    padding: 2rem 1.5rem;
    text-align: center;
    border-radius: 20px;
    background: linear-gradient(145deg, rgba(124, 92, 255, 0.16), rgba(0, 212, 255, 0.07));
    border: 1px solid rgba(255, 255, 255, 0.10);
    backdrop-filter: blur(14px);
    box-shadow: 0 16px 50px rgba(0, 0, 0, 0.40);
    animation: rise 0.35s ease;
}
.result-card .label {
    color: #8b95b3; font-size: 0.78rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.18em;
}
.result-card .flag { font-size: 3.4rem; margin: 0.6rem 0 0.1rem 0; line-height: 1; }
.result-card .lang {
    font-size: 2.4rem; font-weight: 800; letter-spacing: -0.02em;
    background: linear-gradient(90deg, #ffffff, #c4b5fd);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Waiting / hint card */
.hint-card {
    margin-top: 1.5rem; padding: 1.6rem; text-align: center;
    border-radius: 20px; color: #8b95b3;
    border: 1px dashed rgba(255, 255, 255, 0.16);
    background: rgba(255, 255, 255, 0.025);
}

/* Top-k bars */
.bars { margin-top: 1.2rem; }
.bar-row { display: flex; align-items: center; gap: 12px; margin: 10px 0; font-size: 0.92rem; }
.bar-row .name { width: 110px; text-align: left; color: #cdd4ec; font-weight: 500; }
.bar-row .track {
    flex: 1; height: 10px; border-radius: 999px;
    background: rgba(255, 255, 255, 0.07); overflow: hidden;
}
.bar-row .fill {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #7c5cff, #22d3ee);
    transition: width 0.4s ease;
}
.bar-row .fill.dim { background: rgba(255, 255, 255, 0.22); }
.bar-row .pct { width: 52px; text-align: right; color: #8b95b3; font-variant-numeric: tabular-nums; }

/* Expander */
div[data-testid="stExpander"] {
    margin-top: 1.5rem;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.09);
    background: rgba(255, 255, 255, 0.03);
}
div[data-testid="stExpander"] summary p { font-weight: 600; color: #cdd4ec; }

/* Stat chips */
.chips { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px; }
.chip {
    padding: 6px 12px; border-radius: 999px; font-size: 0.82rem; color: #cdd4ec;
    background: rgba(124, 92, 255, 0.14); border: 1px solid rgba(124, 92, 255, 0.35);
}

.footer-note { text-align: center; color: #5d6785; font-size: 0.8rem; margin-top: 2.5rem; }

@keyframes float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }
@keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

@media (max-width: 600px) {
    .hero h1 { font-size: 2rem; }
    .result-card .lang { font-size: 1.9rem; }
    .bar-row .name { width: 80px; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ============================================================
# MODEL
# ============================================================


@st.cache_resource
def load_model():
    return joblib.load("language_model.pkl"), joblib.load("tfidf_vectorizer.pkl")


model, vectorizer = load_model()

# Dataset class names have typos; fix them for display only
DISPLAY_NAMES = {"Portugeese": "Portuguese", "Sweedish": "Swedish"}
FLAGS = {
    "English": "🇬🇧", "French": "🇫🇷", "Spanish": "🇪🇸", "Portuguese": "🇵🇹",
    "Italian": "🇮🇹", "German": "🇩🇪", "Dutch": "🇳🇱", "Swedish": "🇸🇪",
    "Danish": "🇩🇰", "Russian": "🇷🇺", "Greek": "🇬🇷", "Turkish": "🇹🇷",
    "Arabic": "🇸🇦", "Hindi": "🇮🇳", "Tamil": "🇮🇳", "Malayalam": "🇮🇳",
    "Kannada": "🇮🇳",
}


def pretty(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def clean(text: str) -> str:
    # must match train.py exactly
    text = re.sub(r"[\d]", " ", text)
    text = re.sub(r"[!@#$(),\n\"%^*?:;~`\[\]{}]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def detect_top_k(text: str, k: int = 3):
    """Return [(language, relative_share), ...] best-first, or None if input is unusable."""
    text = clean(text or "")
    if len(text) < 3:
        return None

    scores = model.decision_function(vectorizer.transform([text]))[0]
    top = np.argsort(scores)[::-1][:k]

    # Softmax over the SVM margins, only to draw readable bars (not true probabilities)
    logits = scores * 4.0
    exp = np.exp(logits - logits.max())
    share = exp / exp.sum()
    return [(pretty(model.classes_[i]), float(share[i])) for i in top]


# ============================================================
# UI
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="globe">🌍</div>
        <h1>AI Language Detector</h1>
        <p>Start typing and the language is detected live.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

text = st_keyup(
    "Your text",
    placeholder="Type something in any language...",
    debounce=200,
)

results = detect_top_k(text)

if results is None:
    st.markdown(
        '<div class="hint-card">✍️ Waiting for you to type...</div>',
        unsafe_allow_html=True,
    )
else:
    best_lang, _ = results[0]

    bars = "".join(
        f"""
        <div class="bar-row">
            <div class="name">{escape(lang)}</div>
            <div class="track"><div class="fill{' dim' if i else ''}" style="width:{share * 100:.1f}%"></div></div>
            <div class="pct">{share * 100:.0f}%</div>
        </div>
        """
        for i, (lang, share) in enumerate(results)
    )

    st.markdown(
        f"""
        <div class="result-card">
            <div class="label">Detected language</div>
            <div class="flag">{FLAGS.get(best_lang, "🌐")}</div>
            <div class="lang">{escape(best_lang)}</div>
            <div class="bars">{bars}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("ℹ️ About the model"):
    st.markdown(
        """
        <div class="chips">
            <span class="chip">Char TF-IDF</span>
            <span class="chip">n-grams 2–5</span>
            <span class="chip">Linear SVM</span>
            <span class="chip">Accuracy ~99.17%</span>
            <span class="chip">Macro F1 ~99.34%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="footer-note">Built with Streamlit & scikit-learn</div>',
    unsafe_allow_html=True,
)