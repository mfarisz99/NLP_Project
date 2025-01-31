import streamlit as st
from transformers import pipeline

# Load the Hugging Face sentiment analysis pipeline
@st.cache_resource
def load_sentiment_model():
    return pipeline('sentiment-analysis')

# Initialize the sentiment analysis pipeline
sentiment_classifier = load_sentiment_model()

# Set Streamlit page config for custom theme
st.set_page_config(
    page_title="Customer Review Sentiment Analysis",
    page_icon=":speech_balloon:",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for background and styling
st.markdown("""
    <style>
        body {
            background-image: url('https://your-image-url-here.jpg');
            background-size: cover;
            background-position: center;
            color: white;
        }
        .stTextArea textarea {
            background-color: rgba(255, 255, 255, 0.7);
            color: #333;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 12px;
            padding: 10px 24px;
            font-size: 18px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .stTitle {
            font-family: 'Arial', sans-serif;
        }
        .stMarkdown {
            font-family: 'Arial', sans-serif;
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit App UI
st.title("Customer Review Sentiment Analysis")
st.write("This app predicts whether a customer's review is positive or negative.")

# Input text for analysis
user_input = st.text_area("Enter a customer review:", height=100)

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Perform sentiment analysis
        result = sentiment_classifier(user_input)
        text_label = result[0]['label']
        text_score = result[0]['score']

        # Display the result
        if text_label == "POSITIVE":
            st.success(f"Positive Sentiment (Confidence: {text_score:.2f})")
        else:
            st.error(f"Negative Sentiment (Confidence: {text_score:.2f})")
    else:
        st.warning("Please enter a review to analyze.")

# Footer note
st.write("\nModel powered by Hugging Face Transformers.")
