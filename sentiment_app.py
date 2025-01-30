import streamlit as st
from textblob import TextBlob
import pandas as pd

# Title of the app
st.title("Customer Review Sentiment Analysis")

# Input text box for customer review
review = st.text_area("Enter your review here:")

# Analyze sentiment
if st.button("Analyze Sentiment"):
    if review:
        # Perform sentiment analysis using TextBlob
        analysis = TextBlob(review)
        sentiment = analysis.sentiment

        # Display results
        st.write(f"Polarity: {sentiment.polarity}")
        st.write(f"Subjectivity: {sentiment.subjectivity}")

        # Determine sentiment label
        if sentiment.polarity > 0:
            st.success("Positive Sentiment 😊")
        elif sentiment.polarity < 0:
            st.error("Negative Sentiment 😠")
        else:
            st.info("Neutral Sentiment 😐")
    else:
        st.warning("Please enter a review to analyze.")

# Optional: Add a file uploader for bulk analysis
uploaded_file = st.file_uploader("Upload a CSV file with reviews", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.write(data.head())

    if st.button("Analyze Bulk Reviews"):
        results = []
        for index, row in data.iterrows():
            review_text = row["Review"]  # Assuming the column name is "Review"
            analysis = TextBlob(review_text)
            sentiment = analysis.sentiment
            results.append({
                "Review": review_text,
                "Polarity": sentiment.polarity,
                "Subjectivity": sentiment.subjectivity,
                "Sentiment": "Positive" if sentiment.polarity > 0 else "Negative" if sentiment.polarity < 0 else "Neutral"
            })
        
        results_df = pd.DataFrame(results)
        st.write("Sentiment Analysis Results:")
        st.write(results_df)
