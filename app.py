from flask import Flask, render_template, request, send_file
import pandas as pd
import os
import time

from preprocessing import clean_text
from bert_attention_model import predict_sentiment
from explanation import generate_explanation
from fuzzy_logic import get_fuzzy_label
from intuitionistic_fuzzy import compute_intuitionistic_fuzzy

from gcn_model import (
    train_gcn,
    recommend_products_gcn,
    build_graph
)

from graph_visualizer import create_graph_visualization
from metrics import calculate_basic_metrics
from genai_explainer import generate_genai_explanation
from image_analyzer import analyze_uploaded_image

app = Flask(__name__)

print("Starting system...")

# -----------------------------
# LOAD DATASET
# -----------------------------

df = pd.read_csv("amazon_software.csv").head(1000)

total_users = df["user_id"].nunique()
total_products = df["product_id"].nunique()
total_reviews = len(df)

# -----------------------------
# CLEAN TEXT
# -----------------------------

df["clean_review"] = df["review_text"].apply(clean_text)

# -----------------------------
# SENTIMENT
# -----------------------------

def dataset_sentiment(text):

    try:

        sentiment, _ = predict_sentiment(text)

        if sentiment == "Positive":
            return 0.8

        elif sentiment == "Negative":
            return -0.8

        return 0

    except:

        return 0

df["sentiment_score"] = df["clean_review"].apply(dataset_sentiment)

df["sentiment"] = df["sentiment_score"].apply(
    lambda x: "Positive" if x > 0 else "Negative"
)

# -----------------------------
# FUZZY
# -----------------------------

df["fuzzy_strength"] = df["sentiment_score"].apply(get_fuzzy_label)

# -----------------------------
# INTUITIONISTIC
# -----------------------------

fuzzy_values = df["sentiment_score"].apply(
    compute_intuitionistic_fuzzy
)

df["membership"] = fuzzy_values.apply(
    lambda x: x["membership"]
)

df["non_membership"] = fuzzy_values.apply(
    lambda x: x["non_membership"]
)

df["hesitation"] = fuzzy_values.apply(
    lambda x: x["hesitation"]
)

positive_reviews = len(
    df[df["sentiment"] == "Positive"]
)

negative_reviews = len(
    df[df["sentiment"] == "Negative"]
)

# -----------------------------
# TRAIN GCN
# -----------------------------

start_time = time.time()

model, node_index, data = train_gcn(df)

training_time = round(
    time.time() - start_time,
    2
)

G = build_graph(df)

precision, recall, f1 = calculate_basic_metrics()

# Baseline accuracy (for comparison)

baseline_accuracy = 0.82

accuracy_improvement = round(
    (precision - baseline_accuracy) * 100,
    2
)

# -----------------------------
# SAVE IMAGE
# -----------------------------

def save_uploaded_image(file):

    folder = "static/uploads"

    if not os.path.exists(folder):

        os.makedirs(folder)

    filepath = os.path.join(
        folder,
        file.filename
    )

    file.save(filepath)

    return filepath

# -----------------------------
# RECOMMENDATION
# -----------------------------

def get_recommendations_for_user(user_id):

    try:

        recommendations = recommend_products_gcn(
            model,
            node_index,
            data,
            user_id
        )

        df_rec = pd.DataFrame(
            recommendations,
            columns=[
                "product_id",
                "gcn_score"
            ]
        )

        if not df_rec.empty:

            df_rec["gcn_score"] = df_rec[
                "gcn_score"
            ].apply(
                lambda x:
                round(x * 100, 2)
            )

        return df_rec

    except:

        return pd.DataFrame()

@app.route("/", methods=["GET", "POST"])

def index():

    uploaded_image = None
    image_result = None
    image_explanation = None
    image_confidence = None
    extracted_text = None

    default_user = df["user_id"].iloc[0]

    user_id = default_user

    if request.method == "POST":

        user_id = request.form.get(
            "user_id"
        ) or default_user

        image_file = request.files.get(
            "image"
        )

        if image_file:

            uploaded_image = save_uploaded_image(
                image_file
            )

            (
                image_result,
                image_explanation,
                image_confidence,
                extracted_text
            ) = analyze_uploaded_image(
                uploaded_image
            )

    recommended_products = get_recommendations_for_user(
        user_id
    )

    if not recommended_products.empty:

        top_product = recommended_products.iloc[0]["product_id"]

        explanation = generate_explanation(
            df,
            top_product
        )

        score = recommended_products.iloc[0]["gcn_score"]

        confidence = round(
            min(score * 5, 99),
            2
        )

        genai_explanation = generate_genai_explanation(
            top_product,
            score
        )

        create_graph_visualization(
            G,
            recommended_product=top_product
        )

    else:

        explanation = "No recommendations available."

        confidence = 0

        genai_explanation = None

        create_graph_visualization(G)

    table_html = recommended_products.to_html(
        classes="data",
        index=False,
        border=0
    )

    return render_template(

        "index.html",

        user_id=user_id,

        positive=positive_reviews,
        negative=negative_reviews,

        total_users=total_users,
        total_products=total_products,
        total_reviews=total_reviews,

        table_html=table_html,

        explanation=explanation,

        genai_explanation=genai_explanation,

        precision=precision,
        recall=recall,
        f1=f1,

        confidence=confidence,

        baseline_accuracy=baseline_accuracy,

        accuracy_improvement=accuracy_improvement,

        training_time=training_time,

        uploaded_image=uploaded_image,

        image_result=image_result,

        image_explanation=image_explanation,

        image_confidence=image_confidence,

        extracted_text=extracted_text

    )

if __name__ == "__main__":

    app.run(debug=True)