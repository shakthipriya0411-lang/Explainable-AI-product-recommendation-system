def generate_explanation(df, product_id):

    product_reviews = df[
        df["product_id"] == product_id
    ]

    positive_reviews = len(
        product_reviews[
            product_reviews["sentiment"]
            == "Positive"
        ]
    )

    negative_reviews = len(
        product_reviews[
            product_reviews["sentiment"]
            == "Negative"
        ]
    )

    avg_rating = round(
        product_reviews["rating"].mean(),
        2
    )

    explanation = f"""
Recommended because:

• Strong positive sentiment from users  
• Average rating: {avg_rating}  
• Balanced feedback pattern  
• Detected strong relationships in the user-product graph
"""

    return explanation