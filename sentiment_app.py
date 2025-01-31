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
            background: linear-gradient(135deg, #f06, #4a90e2);
            background-size: cover;
            background-position: center;
            color: white;
            font-family: 'Arial', sans-serif;
        }
        .stTextArea textarea {
            background-color: rgba(50, 50, 50, 0.8); /* Darker grey background */
            color: white; /* White text for readability */
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
            padding: 12px;
            transition: all 0.3s ease;
            border: 1px solid #666; /* Slight border for better visibility */
        }
        .stTextArea textarea:focus {
            background-color: rgba(70, 70, 70, 0.9);
            outline: none;
            box-shadow: 0 0 10px rgba(100, 150, 255, 0.5);
        }
        .stButton button {
            background-color: #4C4C4C;
            color: white;
            border-radius: 12px;
            padding: 12px 24px;
            font-size: 18px;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #606060;
            transform: scale(1.05);
        }
        .stTitle {
            font-size: 36px;
            font-weight: 700;
            text-align: center;
            text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.5);
        }
        .stMarkdown {
            font-size: 18px;
            text-align: left; /* Left-aligning the description text */
            font-weight: 500;
        }
        .stAlert {
            margin-top: 20px;
            font-size: 18px;
        }
        .stSuccess, .stError {
            font-size: 20px;
            font-weight: 600;
        }
        .stWarning {
            font-size: 20px;
            font-weight: 600;
            color: #f39c12;
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
