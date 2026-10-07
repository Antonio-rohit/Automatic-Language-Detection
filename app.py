import re

import joblib
import numpy as np
import streamlit as st
from st_keyup import st_keyup

st.set_page_config(page_title="AI Language Detector", page_icon="🌍", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load("language_model.pkl"), joblib.load("tfidf_vectorizer.pkl")


model, vectorizer = load_model()


def clean(text: str) -> str:
    # must match train.py exactly
    text = re.sub(r"[\d]", " ", text)
    text = re.sub(r"[!@#$(),\n\"%^*?:;~`\[\]{}]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def detect_top_k(text: str, k: int = 3):
    """Return [(language, score), ...] sorted best-first, or None for unusable input."""
    text = clean(text or "")
    if len(text) < 3:
        return None

    scores = model.decision_function(vectorizer.transform([text]))[0]
    top = np.argsort(scores)[::-1][:k]
    return [(model.classes_[i], float(scores[i])) for i in top]


st.title("🌍 AI Language Detector")
st.write("Start typing and the language is detected live.")
st.divider()

text = st_keyup(
    "Enter your text",
    placeholder="Start typing here...",
    debounce=200,  # ms to wait after the last keystroke
)

results = detect_top_k(text)

if results is None:
    st.info("✍️ Keep typing...")
else:
    best_lang, best_score = results[0]
    st.success(f"🌐 Detected Language: **{best_lang}**")

    with st.expander("Top predictions"):
        for lang, score in results:
            st.write(f"{lang}: `{score:.2f}`")

with st.expander("ℹ️ About the Model"):
    st.write(
        """
        - Features: character TF-IDF (n-grams 2–5)
        - Classifier: Linear SVM
        - Accuracy: ~99.17% | Macro F1: ~99.34%
        """
    )