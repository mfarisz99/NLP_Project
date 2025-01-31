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
st.write("This app predicts whether a customer's review is positive or negative and visualizes the sentiment strength.")

# Input text for analysis
user_input = st.text_area("Enter a customer review:", height=100)

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Perform sentiment analysis
        result = sentiment_classifier(user_input)
        text_label = result[0]['label']
        text_score = result[0]['score']

        # Normalize sentiment score (0.5 - 1) → (0 - 1)
        normalized_sentiment_score = (text_score - 0.5) * 2  

        # Convert score to percentage (0-100)
        sentiment_percentage = int(normalized_sentiment_score * 100)

        # Define color and status
        if text_label == "POSITIVE":
            sentiment_status = "Positive"
            color = "green"
        else:
            sentiment_status = "Negative"
            color = "red"
            sentiment_percentage = 100 - sentiment_percentage  # Invert for better visualization

        # Display sentiment result
        st.markdown(f"### **Sentiment: {sentiment_status} ({sentiment_percentage}%)**")
        
        # Emotional meter (progress bar)
        st.progress(sentiment_percentage / 100)

    else:
        st.warning("Please enter a review to analyze.")

# Footer note
st.write("\nModel powered by Hugging Face Transformers.")

