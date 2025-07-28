# import streamlit as st
# import tensorflow as tf
# import pickle
# import numpy as np
# import re
# from PIL import Image

# # app.py
# import streamlit as st
# import pickle
# import joblib
# import numpy as np
# import re
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences


# # Constants
# MAX_LEN = 300

# # -------------------------------
# # Load bilstm Model & Tokenizer
# # -------------------------------
# # model = tf.keras.models.load_model("model_bilstm.h5")
# # with open("tokenizer_bilstm.pkl", "rb") as f:
# #     tokenizer = pickle.load(f)

# @st.cache_resource
# def load_artifacts():
#     model = load_model("model_bilstm.h5")
#     with open("tokenizer_bilstm.pkl", "rb") as f:
#         tokenizer = pickle.load(f)
#     label_encoder = joblib.load("label_encoder.pkl")
#     return model, tokenizer, label_encoder

# model, tokenizer, label_encoder = load_artifacts()



# #MAX_LEN = 300  # Same as training    

# # -------------------------------
# # Preprocessing Function
# # -------------------------------
# def preprocess_text(text):
#     text = re.sub(r'http\S+', '', text)
#     text = re.sub(r'[^\w\s]', '', text)
#     text = re.sub(r'Â', '', text)
#     text = re.sub(r'1/2ï', '', text)
#     text = re.sub(r'â', '', text)
#     text = re.sub(r'e mail', '', text)
#     text = re.sub(r'email', '', text)
#     text = re.sub(r'3d', '', text)
#     text = re.sub(r'ï', '', text)
#     text = re.sub(r'hou ', '', text)
#     text = re.sub(r' 000 ', ' ', text)
#     text = re.sub(r' e  ', ' ', text)
#     text = re.sub(r' 00 ', ' ', text)
#     text = re.sub(r' enron ', ' ', text)
#     text = text.lower()
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# # -------------------------------
# # Predict Function
# # -------------------------------
# # def predict_email_bilstm(text):
# #     #cleaned = clean_text(text)
# #     cleaned = preprocess_text(text)

# #     seq = tokenizer.texts_to_sequences([cleaned])
# #     padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=300, padding='post')
# #     pred_prob = model.predict(padded)[0][0]
# #     label = "Phishing" if pred_prob >= 0.5 else "Safe"
# #     confidence = pred_prob if label == "Phishing" else 1 - pred_prob
# #     return label, float(confidence) * 100

# def predict_email(email_text):
#     clean_text = preprocess_text(email_text)
#     seq = tokenizer.texts_to_sequences([clean_text])
#     padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
#     pred_prob = model.predict(padded)[0][0]
#     pred_label = 1 if pred_prob > 0.5 else 0
#     decoded_label = label_encoder.inverse_transform([pred_label])[0]
#     return decoded_label, float(pred_prob)

# # -------------------------------
# # Streamlit Page Config
# # -------------------------------
# #st.set_page_config(page_title="Phishing Email Detector", layout="centered")   # another wide

# st.set_page_config(page_title="Phishing Email Detector", layout="centered")
# st.title("📧 Phishing Email Detection using BiLSTM")
# st.markdown("Enter any email text below to check if it's **Phishing** or **Safe**.")


# # -------------------------------
# # Custom CSS for Header/Footer
# # -------------------------------
# st.markdown("""
#     <style>
#     .main { background-color: #f5f5f5; }
#     header { visibility: hidden; }
#     footer { visibility: hidden; }
#     .custom-header {
#         background-color: #002b36;
#         color: white;
#         padding: 1.5rem;
#         text-align: center;
#         font-size: 32px;
#         border-radius: 10px;
#     }
#     .footer {
#         position: fixed;
#         bottom: 0;
#         width: 100%;
#         text-align: center;
#         font-size: 14px;
#         padding: 10px;
#         background-color: #002b36;
#         color: white;
#     }
#     </style>
# """, unsafe_allow_html=True)

# st.markdown('<div class="custom-header">📧 Smart Phishing Email Detector</div>', unsafe_allow_html=True)

# # -------------------------------
# # Sidebar Navigation
# # -------------------------------
# page = st.sidebar.radio("Navigation", ["Home", "Predict", "About", "Contact", "GitHub", "LinkedIn"])

# # -------------------------------
# # Home Page
# # -------------------------------
# if page == "Home":
#     st.header("🧠 Project Overview")
#     st.markdown("""
#     This project uses a **BILSTM-based deep learning model** to detect phishing emails using Natural Language Processing (NLP).
    
#     - 🔍 **Model:** Gated Recurrent Unit (BILSTM)
#     - 📊 **Data:** Preprocessed labeled email messages
#     - ⚙️ **Tools Used:** Python, TensorFlow, Streamlit
#     - 📦 **Goal:** Help users detect and avoid phishing attempts via emails
    
#     Go to the **Predict** page to test with your own email content.
#     """)

# # -------------------------------
# # Predict Page
# # -------------------------------
# # elif page == "Predict":
# #     st.header("🔐 Email Classification")
# #     st.markdown("Enter email content to detect if it’s **phishing** or **safe**.")

# #     user_input = st.text_area("✉️ Paste the email content here:", height=250)

# #     if st.button("🚀 Predict"):
# #         if user_input.strip() == "":
# #             st.warning("Please enter some email content before prediction.")
# #         else:
# #             label, conf = predict_email_bilstm(user_input)
# #             if label == "Phishing":
# #                 st.error(f"🚨 Prediction: {label} ({conf:.2f}% confidence)")
# #             else:
# #                 st.success(f"✅ Prediction: {label} ({conf:.2f}% confidence)")

# user_input = st.text_area("✉️ Email Text", height=200)

# if st.button("🔍 Predict"):
#     if user_input.strip() == "":
#         st.warning("Please enter some email text.")
#     else:
#         label, confidence = predict_email(user_input)
#         if label.lower() == "phishing":
#             st.error(f"🚨 Prediction: **Phishing** ({confidence:.2f} confidence)")
#         else:
#             st.success(f"✅ Prediction: **Safe** ({confidence:.2f} confidence)")



# -------------------------------
# About Page
# # -------------------------------
# elif page == "About":
#     st.header("ℹ️ About This Project")
#     st.markdown("""
#     This Smart Phishing Email Detector is powered by a deep learning GRU model trained on a dataset of labeled emails.

#     ### 🔧 What It Does
#     - Predicts whether an email is **Phishing** or **Safe**
#     - Works on real-world user input
#     - Helps prevent email fraud and phishing scams

#     ### 🎯 Why This Project Matters
#     - Protects individuals and businesses from phishing attacks
#     - Promotes awareness around cyber threats
#     - Easy-to-use and accessible tool

#     ### 📚 Technologies Used
#     - **TensorFlow**, **Keras**
#     - **Streamlit**
#     - **NLP Preprocessing**
#     """)

# # -------------------------------
# # Contact Page
# # -------------------------------
# elif page == "Contact":
#     st.header("📞 Contact Us")
#     st.markdown("""
#     - **Name:** Adlin Babi  
#     - **Email:** adlinbabi@example.com  
#     - **Location:** India  
#     - **Message:** Feel free to reach out for collaboration or feedback!
#     """)

# # -------------------------------
# # GitHub Page
# # -------------------------------
# elif page == "GitHub":
#     st.header("🌐 GitHub Repository")
#     st.markdown("[🔗 Click here to view the code on GitHub](https://github.com/ADLIN-BABI)")

# # -------------------------------
# # LinkedIn Page
# # -------------------------------
# elif page == "LinkedIn":
#     st.header("💼 LinkedIn Profile")
#     st.markdown("[🔗 Connect on LinkedIn](https://www.linkedin.com/in/adlin-babi-99a123226)")

# # -------------------------------
# # Footer
# # -------------------------------
# st.markdown('<div class="footer">Made with ❤️ by Adlin Babi | Streamlit App | 2025</div>', unsafe_allow_html=True)


import streamlit as st
import numpy as np
import pickle
import joblib
from tensorflow.keras.models import load_model  # ✅ Fixes the error
from tensorflow.keras.preprocessing.sequence import pad_sequences



import tensorflow as tf
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from streamlit_option_menu import option_menu

# -----------------------------
# Load Model, Tokenizer, Label Encoder
# -----------------------------

st.title("📨 Predict Email Type (GRU Model)")

# Load GRU model and preprocessing tools
# Later on, to load the model and tokenizer:
# model_gru_loaded = load_model("model_gru.h5")
# with open("tokenizer_gru.pkl", "rb") as f:
#     tokenizer = pickle.load(f)

# with open("label_encoder.pkl", "rb") as f:
#     label_encoder = pickle.load(f)


# label_encoder = LabelEncoder()
# label_encoder.fit(["phishing","safe"]) 
# joblib.dump(label_encoder, 'label_encoder.pkl')

# Later on, to load the model and tokenizer:
model_gru_loaded = load_model("model_gru.h5")


with open("tokenizer_gru.pkl", "rb") as f_tok:
    tokenizer_gru_loaded = pickle.load(f_tok)

label_encoder_loaded = joblib.load('label_encoder.pkl')

MAX_LEN = 150


# -----------------------------
# Utility Functions
# -----------------------------
# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r"http\S+|www.\S+", "", text)
#     text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
#     text = re.sub(r"\s+", " ", text).strip()
#     return text

# def predict_email(email):
#     clean = preprocess_text(email)
#     seq = tokenizer_gru_loaded.texts_to_sequences([clean])
#     padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
#     pred = model_gru_loaded.predict(padded)[0][0]
#     label = 0 if pred >= 0.5 else 1
#     label_str = label_encoder_loaded.inverse_transform([label])[0]
#     return label_str, float(pred)

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
        # Convert numeric label to string label
#         try:
#             label_str = label_encoder.inverse_transform([predicted_label])[0].lower()
#         except Exception as e:
#            st.error(f"Error decoding label: {e}")
#            label_str = "unknown"


        # Display result
        if label_str.lower() == "phishing":
            st.error("🚨 This email is predicted to be: PHISHING")
        else:
            st.success("✅ This email is predicted to be: SAFE")
#     email_input = st.text_area("Enter the email content below:")

# if st.button("Classify"):
#     if not email_input.strip():
#         st.warning("Please enter an email to classify.")
#     else:
#         label, confidence = predict_email(email_input)
#         if label.lower() == "phishing":
#             st.error(f"🚨 This is a **Phishing Email**.\nConfidence: {confidence:.2f}")
#         else:
#             st.success(f"✅ This is a **Safe Email**.\nConfidence: {confidence:.2f}")

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

