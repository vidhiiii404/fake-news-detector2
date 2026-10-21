import joblib
import numpy as np
import pandas as pd
import streamlit as st

from text_utils import clean_text  # needed so joblib can load the pipeline

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load("fake_news_model.joblib")


model = load_model()


def p_real(text: str) -> float:
    x = [clean_text(text)]
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(x)[0][1])
    return float(1 / (1 + np.exp(-model.decision_function(x)[0])))


st.title("📰 Fake News Detector")
st.caption("NLP project (TF-IDF + machine learning) trained on the FakeNewsNet / GossipCop headline dataset.")

tab1, tab2 = st.tabs(["Check a headline", "About the model"])

with tab1:
    examples = {
        "-- choose an example --": "",
        "Celebrity birthday": "Liev Schreiber and Naomi Watts celebrate son's 10th birthday together",
        "Tabloid style": "Kim Kardashian secretly pregnant with alien baby, insiders reveal",
        "Award news": "People's Choice Awards 2018: the best red carpet looks",
    }
    choice = st.selectbox("Try an example", list(examples))
    text = st.text_area("News headline", value=examples[choice], height=100,
                        placeholder="Paste a news headline here...")

    if st.button("Analyze", type="primary"):
        if len(text.strip()) < 5:
            st.warning("Please enter a longer headline.")
        else:
            pr = p_real(text)
            if pr >= 0.5:
                st.success(f"✅ Likely REAL  (confidence {pr:.0%})")
            else:
                st.error(f"🚨 Likely FAKE  (confidence {1 - pr:.0%})")
            st.progress(pr, text=f"Probability real: {pr:.1%}")
            st.caption("This is a statistical prediction from wording alone, not a fact-check.")

with tab2:
    st.write("**Pipeline:** text cleaning → TF-IDF (unigrams + bigrams) → classifier chosen after comparing four models.")
    try:
        st.dataframe(pd.read_csv("model_comparison.csv"), width="stretch")
    except FileNotFoundError:
        st.info("model_comparison.csv not found.")
    st.write("**Limitations:** headline-only input, English gossip/entertainment domain, imbalanced training data (75% real).")
