import matplotlib.pyplot as plt

def plot_fuzzy_values(
    membership,
    non_membership,
    hesitation
):

    labels = [
        "Membership",
        "Non-membership",
        "Hesitation"
    ]

    values = [
        membership,
        non_membership,
        hesitation
    ]

    plt.figure()

    plt.bar(labels, values)

    plt.title(
        "Intuitionistic Fuzzy Values"
    )

    plt.savefig(
        "static/fuzzy_chart.png"
    )

    plt.close()