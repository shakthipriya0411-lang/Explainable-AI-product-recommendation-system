def generate_genai_explanation(
    product_id,
    score
):

    explanation = f"""
    This product is recommended because
    the graph neural network detected
    strong relationships between this
    product and users with similar
    preferences.

    Recommendation strength score:
    {score}
    """

    return explanation