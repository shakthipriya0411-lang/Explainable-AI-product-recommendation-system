import pandas as pd
from transformers import pipeline
from recommender import generate_recommendations
from explanation import generate_explanation
from preprocessing import clean_text

print("Loading Dataset...\n")

# Load dataset FIRST
df = pd.read_csv("amazon_software.csv")

print("Dataset Loaded Successfully!")
print("Total Reviews:", len(df))

print("\nDataset Preview:")
print(df.head())

print("\nDataset Shape:", df.shape)

# -----------------------------
# TEXT PREPROCESSING
# -----------------------------
print("\nCleaning Review Text...\n")

df['clean_review'] = df['review_text'].apply(clean_text)

# -----------------------------
# BERT SENTIMENT MODEL
# -----------------------------
print("\nLoading BERT Sentiment Model...\n")

sentiment_model = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    try:
        result = sentiment_model(text[:512])[0]
        label = result['label']
        score = result['score']

        if label == "POSITIVE":
            return score
        else:
            return -score
    except:
        return 0

print("\nPerforming Sentiment Analysis...\n")

df['sentiment_score'] = df['clean_review'].apply(analyze_sentiment)

df['sentiment'] = df['sentiment_score'].apply(
    lambda x: "Positive" if x > 0 else "Negative"
)

print(df[['review_text','sentiment','sentiment_score']].head())

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
print("\nExtracting Features...\n")

df['review_length'] = df['review_text'].apply(lambda x: len(str(x).split()))
df['sentiment_strength'] = df['sentiment_score'].abs()

# -----------------------------
# RECOMMENDATION
# -----------------------------
print("\nGenerating Product Recommendations...\n")

recommended_products = generate_recommendations(df)

print("Top 5 Recommended Products:")
print(recommended_products.head())

# -----------------------------
# EXPLANATION
# -----------------------------
top_product = recommended_products.iloc[0]['product_id']

print("\nBest Product:", top_product)
print("Explanation:", generate_explanation(df, top_product))