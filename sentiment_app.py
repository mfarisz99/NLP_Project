
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
st.write("This app predicts whether a customer's review is positive or negative, factoring in your satisfaction ratings.")

# Satisfaction Ratings

quality_rating = st.radio("Quality of the item:", ['⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'], index=2)
functionality_rating = st.radio("Functionality of the item:", ['⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'], index=2)
price_worthiness_rating = st.radio("Is the price worth it?", ['⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'], index=2)


st.subheader("Rate Your Satisfaction")
quality_rating = st.selectbox("Quality of the item:", ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"], index=2)
functionality_rating = st.selectbox("Functionality of the item:", ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"], index=2)
price_worthiness_rating = st.selectbox("Is the price worth it?", ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"], index=2)

# Convert star ratings to numerical values
rating_mapping = {"⭐": 1, "⭐⭐": 2, "⭐⭐⭐": 3, "⭐⭐⭐⭐": 4, "⭐⭐⭐⭐⭐": 5}
quality_score = rating_mapping[quality_rating]
functionality_score = rating_mapping[functionality_rating]
price_score = rating_mapping[price_worthiness_rating]

# Input text for analysis
user_input = st.text_area("Enter a customer review:", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Perform sentiment analysis
        result = sentiment_classifier(user_input)
        text_label = result[0]['label']
        text_score = result[0]['score']

        # Calculate average star rating score (normalized to 0-1 scale)
        average_star_score = (quality_score + functionality_score + price_score) / 15

        # Final sentiment adjustment
        adjusted_score = (text_score * 0.7) + (average_star_score * 0.3)
        final_label = "POSITIVE" if adjusted_score >= 0.5 else "NEGATIVE"

        # Display the result
        if final_label == "POSITIVE":
            st.success(f"Positive Sentiment (Adjusted Confidence: {adjusted_score:.2f})")
        else:
            st.error(f"Negative Sentiment (Adjusted Confidence: {adjusted_score:.2f})")
    else:
        st.warning("Please enter a review to analyze.")

# Footer note
st.write("\nModel powered by Hugging Face Transformers.")

