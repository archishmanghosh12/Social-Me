import pandas as pd
import nltk
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer

# Download necessary NLTK data
nltk.download("vader_lexicon")
nltk.download("punkt")

# Initialize SentimentIntensityAnalyzer (VADER for social media text)
sia = SentimentIntensityAnalyzer()

# Load DistilBERT sentiment analysis model (Hugging Face)
sentiment_model = pipeline("sentiment-analysis")

# Function to apply VADER sentiment analysis
def get_vader_sentiment(text):
    score = sia.polarity_scores(text)
    return score["compound"]  # Compound score (-1 to 1)

# Function to apply DistilBERT sentiment analysis
def get_bert_sentiment(text):
    result = sentiment_model(text)[0]
    return result["label"], result["score"]

# Get dynamic input from the user
num_inputs = int(input("Enter the number of text inputs for sentiment analysis: "))
texts = [input(f"Enter text {i+1}: ") for i in range(num_inputs)]

# Create DataFrame
df = pd.DataFrame({"text": texts})

# Apply VADER sentiment
df["vader_sentiment"] = df["text"].apply(get_vader_sentiment)

# Apply DistilBERT model
df["bert_sentiment"], df["bert_confidence"] = zip(*df["text"].apply(get_bert_sentiment))

# Display results
print("\nSentiment Analysis Results:")
print(df)

# Visualizing sentiment results
plt.figure(figsize=(10, 5))
sns.histplot(df["vader_sentiment"], bins=10, kde=True, color="blue", alpha=0.6)
plt.title("Sentiment Distribution (VADER)")
plt.xlabel("Sentiment Score (-1 to 1)")
plt.ylabel("Frequency")
plt.show()

