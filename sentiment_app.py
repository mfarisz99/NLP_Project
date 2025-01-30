import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Title of the app
st.title("Customer Review Sentiment Analysis")

# Input text box for customer review
review = st.text_area("Enter your review here:")

# Analyze sentiment
if st.button("Analyze Sentiment"):
    if review:
        # Perform sentiment analysis using VADER
        sentiment = analyzer.polarity_scores(review)

        # Display results
        st.write(f"Positive: {sentiment['pos']}")
        st.write(f"Neutral: {sentiment['neu']}")
        st.write(f"Negative: {sentiment['neg']}")
        st.write(f"Compound Score: {sentiment['compound']}")

        # Determine sentiment label
        if sentiment['compound'] >= 0.05:
            st.success("Positive Sentiment 😊")
        elif sentiment['compound'] <= -0.05:
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
            sentiment = analyzer.polarity_scores(review_text)
            results.append({
                "Review": review_text,
                "Positive": sentiment['pos'],
                "Neutral": sentiment['neu'],
                "Negative": sentiment['neg'],
                "Compound Score": sentiment['compound'],
                "Sentiment": "Positive" if sentiment['compound'] >= 0.05 else "Negative" if sentiment['compound'] <= -0.05 else "Neutral"
            })
        
        results_df = pd.DataFrame(results)
        st.write("Sentiment Analysis Results:")
        st.write(results_df)
