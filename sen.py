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
    score = sia.polarity_scores(str(text))  # Convert to string to avoid NaN issues
    return score["compound"]  # Compound score (-1 to 1)

# Function to apply DistilBERT sentiment analysis
def get_bert_sentiment(text):
    result = sentiment_model(str(text))[0]  # Convert to string
    return result["label"], result["score"]

# Function to read CSV and process sentiment analysis
def process_csv(file_path):
    df = pd.read_csv(file_path, encoding="utf-8")

    # Ensure the "text" column exists
    if "text" not in df.columns:
        raise ValueError(f"CSV file {file_path} must contain a 'text' column.")

    # Drop any empty text values
    df = df.dropna(subset=["text"])

    # Apply Sentiment Analysis
    df["vader_sentiment"] = df["text"].apply(get_vader_sentiment)
    df["bert_sentiment"], df["bert_confidence"] = zip(*df["text"].apply(get_bert_sentiment))

    return df

# Get CSV file paths from the user
file1 = input("Enter the path of the first CSV file: ")
file2 = input("Enter the path of the second CSV file: ")

# Process both files
df1 = process_csv(file1)
df2 = process_csv(file2)

# Combine both DataFrames (after sentiment analysis)
df_combined = pd.concat([df1, df2], ignore_index=True)

# Display results
print("\nSentiment Analysis Results:")
print(df_combined.head())

# Visualizing sentiment results
plt.figure(figsize=(10, 5))
sns.histplot(df_combined["vader_sentiment"], bins=10, kde=True, color="blue", alpha=0.6)
plt.title("Sentiment Distribution (VADER)")
plt.xlabel("Sentiment Score (-1 to 1)")
plt.ylabel("Frequency")
plt.show()