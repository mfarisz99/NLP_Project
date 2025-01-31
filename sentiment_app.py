import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from transformers import pipeline

# Load the Hugging Face sentiment analysis pipeline
@st.cache_resource
def load_sentiment_model():
    return pipeline('sentiment-analysis')

# Initialize the sentiment analysis pipeline
sentiment_classifier = load_sentiment_model()

# Function to create the emotional meter
def draw_meter(sentiment_score):
    fig, ax = plt.subplots(figsize=(4, 2))

    # Define gauge sections
    colors = ['red', 'orange', 'green']
    labels = ["Negative 😞", "Neutral 😐", "Positive 😊"]
    start = [-np.pi/2, -np.pi/6, np.pi/6]
    end = [-np.pi/6, np.pi/6, np.pi/2]

    # Draw gauge sections
    for i in range(3):
        ax.plot([0, np.cos(start[i])], [0, np.sin(start[i])], color='black', lw=2)
        ax.plot([0, np.cos(end[i])], [0, np.sin(end[i])], color='black', lw=2)
        ax.fill_betweenx([0, 1], np.cos(start[i]), np.cos(end[i]), color=colors[i], alpha=0.6, linewidth=0)

    # Convert sentiment score (0-1) to angle (-90° to 90°)
    angle = -90 + (sentiment_score * 180)
    angle_rad = np.deg2rad(angle)

    # Draw meter needle
    ax.plot([0, np.cos(angle_rad) * 0.8], [0, np.sin(angle_rad) * 0.8], color='black', lw=3)
    ax.scatter(0, 0, color='black', s=50)  # Needle pivot point

    # Hide axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-0.2, 1])

    return fig

# Streamlit UI
st.title("Customer Review Sentiment Analysis")
st.write("This app predicts whether a customer's review is **positive, neutral, or negative** and visualizes it using an emotional meter.")

# User input
user_input = st.text_area("Enter a customer review:", height=100)

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Perform sentiment analysis
        result = sentiment_classifier(user_input)
        text_label = result[0]['label']
        text_score = result[0]['score']

        # Normalize sentiment score (0.5 - 1 → 0 - 1 scale)
        normalized_sentiment_score = (text_score - 0.5) * 2  

        # Generate and display emotional meter
        fig = draw_meter(normalized_sentiment_score)
        st.pyplot(fig)

        # Display sentiment result
        st.markdown(f"### **Sentiment: {text_label} ({normalized_sentiment_score:.2f})**")

    else:
        st.warning("Please enter a review to analyze.")

# Footer
st.write("\nModel powered by Hugging Face Transformers.")
