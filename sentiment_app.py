import streamlit as st
from transformers import pipeline

# Load the Hugging Face sentiment analysis pipeline
@st.cache_resource
def load_sentiment_model():
    return pipeline('sentiment-analysis')

# Initialize the sentiment analysis pipeline
sentiment_classifier = load_sentiment_model()

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
