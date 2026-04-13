import pandas as pd

def generate_recommendations(df):

    df_group = df.groupby(
        "product_id"
    ).agg({

        "sentiment_score": "mean",

        "rating": "mean",

        "review_text": "count",

        "review_length": "mean",

        "sentiment_strength": "mean"

    }).reset_index()

    df_group.rename(columns={

        "review_text": "review_count"

    }, inplace=True)

    df_group["final_score"] = (

        df_group["sentiment_score"] * 0.3 +

        df_group["rating"] * 0.2 +

        df_group["review_count"] * 0.2 +

        df_group["sentiment_strength"] * 0.2 +

        df_group["review_length"] * 0.1

    )

    recommendations = df_group.sort_values(

        by="final_score",

        ascending=False

    )

    return recommendations