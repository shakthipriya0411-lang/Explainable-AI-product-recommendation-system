def get_fuzzy_label(score):

    if score >= 0.75:
        return "Strong Positive"

    elif score >= 0.5:
        return "Weak Positive"

    elif score >= 0:
        return "Neutral"

    elif score >= -0.5:
        return "Weak Negative"

    else:
        return "Strong Negative"