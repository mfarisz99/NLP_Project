import streamlit as st
from transformers import pipeline

# Load the Hugging Face sentiment analysis pipeline
@st.cache_resource
def load_sentiment_model():
    return pipeline('sentiment-analysis')

# Initialize the sentiment analysis pipeline
sentiment_classifier = load_sentiment_model()

# Streamlit App UI
st.title("Customer Review Sentiment Analysis 😊😐😡")
st.write("This app predicts the sentiment of your review and visualizes it with an emotional face.")

# Input text for analysis
user_input = st.text_area("Enter a customer review:", height=100)

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Perform sentiment analysis
        result = sentiment_classifier(user_input)
        text_label = result[0]['label']
        text_score = result[0]['score']

        # Normalize sentiment score (convert 0.5-1 to 0-1)
        normalized_sentiment_score = (text_score - 0.5) * 2  

        # Determine sentiment category
        if normalized_sentiment_score >= 0.7:
            final_label = "POSITIVE"
            emoji = "😊"
            color = "green"
        elif normalized_sentiment_score >= 0.3:
            final_label = "NEUTRAL"
            emoji = "😐"
            color = "orange"
        else:
            final_label = "NEGATIVE"
            emoji = "😡"
            color = "red"

        # Display the result with an emotional face
        st.markdown(f"<h1 style='text-align: center; color: {color};'>{emoji}</h1>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center; color: {color};'>{final_label} (Confidence: {text_score:.2f})</h3>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a review to analyze.")

# Footer note
st.write("\nModel powered by Hugging Face Transformers.")
