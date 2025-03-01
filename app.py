import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import re

# Download necessary NLTK resources (cached)
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords')
    nltk.download('wordnet')

download_nltk_data()

# Load stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Custom tokenizer to avoid NLTK punkt issues
def custom_tokenize(text):
    return re.split(r'\W+', text)  # Splitting on non-word characters

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    tokens = custom_tokenize(text)  # Use regex-based tokenizer instead of word_tokenize
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load saved model and TF-IDF vectorizer
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Define label mapping dictionary (use the same mapping from training)
label_mapping = {0: "negative", 1: "neutral", 2: "positive"}

# Streamlit UI with logo
st.set_page_config(page_title="Sentiment Classifier", page_icon="🤖")
st.image("https://myhotposters.com/cdn/shop/products/mNS0022_1024x1024.jpeg?v=1571444033", use_container_width=True)

st.title("Multiclass Sentiment Classification Prediction Model")

user_input = st.text_area("Please enter a sentence to classify.")

if st.button("Predict!"):
    processed = preprocess_text(user_input)
    vectorized = tfidf.transform([processed])
    prediction_encoded = model.predict(vectorized)[0]  # Get numerical prediction
    prediction_proba = model.predict_proba(vectorized)  # Get confidence scores
    confidence = np.max(prediction_proba) * 100  # Get highest confidence percentage

    # Convert encoded prediction (0,1,2) to human-readable sentiment label
    prediction_label = label_mapping[prediction_encoded]

    st.write(f"**Predicted Sentiment**: {prediction_label}")
    st.write(f"**Confidence**: {confidence:.2f}%")

# Credit section
st.markdown("---")
st.markdown("**Build by Li Fan, 2025-02-28**")
