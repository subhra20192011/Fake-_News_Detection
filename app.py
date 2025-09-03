import streamlit as st
import joblib

# Load model + vectorizer
model = joblib.load("log_reg_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Set up background image using CSS
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
  background-image: url("https://i.pinimg.com/736x/6c/bb/36/6cbb36a34afe45944dc1280aed5f9c0c.jpg");
  background-size: cover;
  background-repeat: no-repeat;
  background-attachment: fixed;
  background-position: center;
}}

[data-testid="stHeader"] {{
  background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
  right: 2rem;
}}
</style>
"""



st.markdown(page_bg_img, unsafe_allow_html=True)

# App title (centered & styled for contrast)
st.markdown(
    "<h1 style='text-align: center; color: white; text-shadow: 2px 2px #000;'>📰📰 Fake News Detection 📰📰</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center; color: white; text-shadow: 2px 2px #000;'>Enter Your Text/News Below 👇</h4>",
    unsafe_allow_html=True
)

# Input for news text
user_input = st.text_area(
    label="",
    height=200,
    key="news_input"
)

# Prediction section
if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        tfidf_input = vectorizer.transform([user_input])
        pred = model.predict(tfidf_input)[0]

        # Get model confidence
        try:
            proba = model.predict_proba(tfidf_input)[0]
            confidence = proba[pred] * 100
        except AttributeError:
            confidence = None

        if pred == 1:
            st.error(f"🚨 **Fake News** (confidence: {confidence:.2f}%)" if confidence is not None else "🚨 **Fake News**")
        else:
            st.success(f"✅ **Real News** (confidence: {confidence:.2f}%)" if confidence is not None else "✅ **Real News**")
