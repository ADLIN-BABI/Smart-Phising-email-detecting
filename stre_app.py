
import streamlit as st
import tensorflow as tf
import numpy as np
import re
import pickle
import joblib

from tensorflow.keras.models import load_model  # Fixes the error
from tensorflow.keras.preprocessing.sequence import pad_sequences
from streamlit_option_menu import option_menu



# -----------------------------
# Load Model, Tokenizer, Label Encoder
# -----------------------------

st.title("📨 Predict Email Type (GRU Model)")



# Later on, to load the model and tokenizer:
model_gru_loaded = load_model("model_gru.h5")


with open("tokenizer_gru.pkl", "rb") as f_tok:
    tokenizer_gru_loaded = pickle.load(f_tok)

label_encoder_loaded = joblib.load('label_encoder.pkl')

MAX_LEN = 150


# -----------------------------
# Streamlit Page Config (No Background)
# -----------------------------
st.set_page_config(
    page_title="Phishing Email Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Sidebar Navigation
# -----------------------------
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Home", "Predict", "About", "Contact Us", "GitHub", "LinkedIn"],
        icons=["house", "envelope-open", "info-circle", "envelope", "github", "linkedin"],
        menu_icon="cast",
        default_index=0,
    )

# -----------------------------
# Pages
# -----------------------------
if selected == "Home":
    st.title("📧 Smart Phishing Email Detector")
    st.write("This web app detects phishing emails using a deep learning BiLSTM model.")
    st.markdown("Enter your email on the **Predict** page to get instant classification as `Safe` or `Phishing`.")

elif selected == "Predict":
    st.title("🔍 Predict Email Safety")
    email_input = st.text_area("Email Content", height=300)

if st.button("Predict"):
    if email_input.strip() == "":
        st.warning("Please enter some email content to classify.")
    else:
        # Preprocess input
        sequence = tokenizer_gru_loaded.texts_to_sequences([email_input])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN)

        # Make prediction
        prediction = model_gru_loaded.predict(padded_sequence)[0][0]
        label = 0 if prediction >= 0.5 else 1
        label_str = label_encoder_loaded.inverse_transform([label])[0]
      


        # Display result
        if label_str.lower() == "phishing":
            st.error("🚨 This email is predicted to be: PHISHING")
        else:
            st.success("✅ This email is predicted to be: SAFE")


elif selected == "About":
    st.title("📘 About")
    st.write("""
        This project uses a Bidirectional LSTM deep learning model to detect phishing emails.  
        It preprocesses the text, tokenizes it, and classifies it as either:
        - **Phishing**
        - **Safe**

        **Technologies Used:**  
        - Python, TensorFlow  
        - BiLSTM, NLP  
        - Streamlit (for web app)  
    """)

elif selected == "Contact Us":
    st.title("📬 Contact Us")
    st.markdown("**Name:** Adlin Babi  \n**Email:** adlin@example.com  \n**Project:** Smart Email Phishing Detection")

elif selected == "GitHub":
    st.title("🔗 GitHub")
    st.markdown("[Visit GitHub Repo](https://github.com/your-repo-link)")

elif selected == "LinkedIn":
    st.title("🔗 LinkedIn")
    st.markdown("[Visit LinkedIn Profile](https://www.linkedin.com/in/your-profile)")

