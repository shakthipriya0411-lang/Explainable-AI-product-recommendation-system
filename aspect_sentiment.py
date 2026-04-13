from bert_attention_model import predict_sentiment
from aspect_extractor import extract_aspects

def analyze_aspect_sentiment(review):

    aspects = extract_aspects(review)

    sentiment, important_words = predict_sentiment(review)

    aspect_results = {}

    for aspect in aspects:
        aspect_results[aspect] = sentiment

    return aspect_results, important_words