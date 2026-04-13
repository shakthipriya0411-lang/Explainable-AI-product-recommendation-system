def calculate_confidence(score):

    """
    Convert GCN score into confidence percentage.
    """

    if score is None:
        return 0

    confidence = min(
        max(score * 60000, 50),
        99
    )

    return round(confidence, 2)