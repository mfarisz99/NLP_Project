import streamlit as st
from transformers import pipeline
import numpy as np

# Initialize sentiment analysis pipeline
classifier = pipeline('sentiment-analysis')

# App Title
st.title("Customer Review Sentiment Analysis")
st.write("Enter any customer review to analyze the sentiment. The result will be displayed as a meter.")

# User Input
user_input = st.text_area("Enter a customer review:")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Analyze the sentiment
        result = classifier(user_input)
        sentiment = result[0]['label']
        score = result[0]['score']

        # Normalize score to percentage
        sentiment_score = np.round(score * 100, 2)

        # Define meter ranges and labels
        if sentiment == 'POSITIVE':
            sentiment_meter = f"Positive Sentiment ({sentiment_score}%)"
            meter_color = "green"
        elif sentiment == 'NEGATIVE':
            sentiment_meter = f"Negative Sentiment ({sentiment_score}%)"
            meter_color = "red"
        else:
            sentiment_meter = f"Neutral Sentiment ({sentiment_score}%)"
            meter_color = "gray"
        
        # Display Sentiment Result
        st.metric(label="Sentiment Analysis Result", value=sentiment_meter)
        
        # Create Custom Sentiment Gauge
        st.progress(int(score * 100))

    else:
        st.warning("Please enter a review before clicking Analyze.")
