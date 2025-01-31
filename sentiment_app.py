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

st.subheader("Rate the Quality of the Item:")
# Define star ratings
stars = ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"]
rating_mapping = {1: "⭐", 2: "⭐⭐", 3: "⭐⭐⭐", 4: "⭐⭐⭐⭐", 5: "⭐⭐⭐⭐⭐"}

# Use st.columns for a better UI layout
cols = st.columns(5)

# Initialize session state for rating
if "product_score" not in st.session_state:
    st.session_state["product_score"] = 3  # Default to 3-star rating

# Button-based star selection
for i, col in enumerate(cols, start=1):
    if col.button(stars[i - 1]):
        st.session_state["product_score"] = i

# Display selected rating
st.markdown(f"### Selected Rating: {rating_mapping[st.session_state['product_score']]}")

# Input text for analysis
user_input = st.text_area("Enter a customer review:", height=100)

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Perform sentiment analysis
        result = sentiment_classifier(user_input)
        text_label = result[0]['label']
        text_score = result[0]['score']

        # Normalize sentiment score (convert from 0.5-1 → 0-1 scale)
        normalized_sentiment_score = (text_score - 0.5) * 2 

        # Convert star rating to numerical score (normalized to 0-1 scale)
        product_score = st.session_state["product_score"]
        star_score = (product_score - 1) / 4  # Normalize to range 0-1
        
        # Adjust weight dynamically based on sentiment confidence
        if text_label == "POSITIVE":
            adjusted_score = (normalized_sentiment_score * 0.6) + (star_score * 0.4)
        else:
            adjusted_score = (normalized_sentiment_score * 0.7) + (star_score * 0.3)  # Give negative review more weight

        final_label = "POSITIVE" if adjusted_score >= 0.6 else "NEGATIVE" 

        # Display the result
        if final_label == "POSITIVE":
            st.success(f"Positive Sentiment (Adjusted Confidence: {adjusted_score:.2f})")
        else:
            st.error(f"Negative Sentiment (Adjusted Confidence: {adjusted_score:.2f})")
    else:
        st.warning("Please enter a review to analyze.")

# Footer note
st.write("\nModel powered by Hugging Face Transformers.")

