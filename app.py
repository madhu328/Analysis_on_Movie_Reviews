"""Interactive sentiment prediction and side-by-side model comparison."""
from pathlib import Path
import streamlit as st
from movie_sentiment.predict import available_models

st.set_page_config(page_title="IMDb Movie Review Sentiment", page_icon="🎬", layout="wide")
st.title("🎬 Movie Review Sentiment Analysis")
st.write("Compare traditional TF-IDF, recurrent neural networks, and transformers on a review.")
models = available_models()
if not models:
    st.info("No trained checkpoints found in artifacts/. Train at least one model using the README instructions, then reload this page.")
    st.stop()

review = st.text_area("Enter a movie review", value="The performances were excellent, but the plot was disappointing.", height=150)
selected = st.multiselect("Choose trained models", options=list(models), default=list(models))
if st.button("Analyze sentiment", type="primary"):
    if not review.strip():
        st.error("Please enter some review text.")
    elif not selected:
        st.error("Please select at least one model.")
    else:
        for name in selected:
            with st.spinner(f"Running {name}..."):
                try:
                    result = models[name](review)
                    st.subheader(name)
                    a, b = st.columns(2)
                    a.metric("Predicted sentiment", result["label"].title())
                    b.metric("Positive probability", f"{result['positive_probability']:.1%}")
                    st.progress(result["positive_probability"])
                except Exception as exc:
                    st.error(f"Could not load {name}: {exc}")

comparison = Path("artifacts/model_comparison.csv")
if comparison.exists():
    import pandas as pd
    st.subheader("Held-out test-set comparison")
    st.dataframe(pd.read_csv(comparison), use_container_width=True)
else:
    st.caption("Model-comparison results will appear after training and running `imdb-sentiment compare`.")