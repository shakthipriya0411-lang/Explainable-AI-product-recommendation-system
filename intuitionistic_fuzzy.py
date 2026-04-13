import numpy as np

# -----------------------------
# INTUITIONISTIC FUZZY LOGIC
# -----------------------------

def compute_intuitionistic_fuzzy(sentiment_score):

    """
    Input:
        sentiment_score: value between -1 and 1

    Output:
        membership (μ)
        non_membership (ν)
        hesitation (π)
    """

    # Normalize sentiment to [0,1]
    normalized = (sentiment_score + 1) / 2

    # Membership
    membership = normalized

    # Non-membership
    non_membership = 1 - membership

    # Hesitation / uncertainty
    hesitation = 1 - membership - non_membership

    # Safety clamp
    membership = max(0, min(1, membership))
    non_membership = max(0, min(1, non_membership))
    hesitation = max(0, min(1, hesitation))

    return {
        "membership": membership,
        "non_membership": non_membership,
        "hesitation": hesitation
    }