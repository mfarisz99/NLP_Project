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

# Satisfaction Ratings
st.subheader("Rate Your Satisfaction")
quality_rating = st.slider("Quality of the item:", 1, 5, 3, format="%d star(s)")
functionality_rating = st.slider("Functionality of the item:", 1, 5, 3, format="%d star(s)")
price_worthiness_rating = st.slider("Is the price worth it?", 1, 5, 3, format="%d star(s)")

st.write("Your Ratings:")
st.write(f"- Quality: {quality_rating} star(s)")
st.write(f"- Functionality: {functionality_rating} star(s)")
st.write(f"- Price Worthiness: {price_worthiness_rating} star(s)")

# Input text for analysis
user_input = st.text_area("Enter a customer review:", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Perform sentiment analysis
        result = sentiment_classifier(user_input)
        label = result[0]['label']
        score = result[0]['score']

        # Display the result
        if label == "POSITIVE":
            st.success(f"Positive Sentiment (Confidence: {score:.2f})")
        else:
            st.error(f"Negative Sentiment (Confidence: {score:.2f})")
    else:
        st.warning("Please enter a review to analyze.")

# Footer note
st.write("\nModel powered by Hugging Face Transformers.")
