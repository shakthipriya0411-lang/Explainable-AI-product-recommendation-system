import os
import json
import pytesseract
from PIL import Image
import re

# -----------------------------
# CONFIG
# -----------------------------

STATS_FILE = "error_stats.json"

# Expanded keyword detection

ERROR_KEYWORDS = {
    "installation": [
        "installation",
        "installer",
        "setup",
        "package"
    ],

    "network": [
        "network",
        "connection",
        "timeout",
        "internet"
    ],

    "login": [
        "login",
        "password",
        "authentication"
    ],

    "permission": [
        "permission",
        "administrator",
        "access denied"
    ],

    "update": [
        "update",
        "upgrade",
        "patch"
    ],

    "crash": [
        "crash",
        "failed",
        "unexpected",
        "exception"
    ]
}

# Solutions

SOLUTIONS = {
    "installation": [
        "Re-download the installer",
        "Run the installer as administrator",
        "Check system compatibility"
    ],

    "network": [
        "Check internet connection",
        "Restart router",
        "Disable firewall temporarily"
    ],

    "login": [
        "Verify username and password",
        "Reset password",
        "Check account permissions"
    ],

    "permission": [
        "Run program as administrator",
        "Check user privileges"
    ],

    "update": [
        "Restart system",
        "Check update server",
        "Install latest version"
    ],

    "crash": [
        "Restart application",
        "Update software",
        "Check system memory"
    ],

    "general": [
        "Restart the application",
        "Check software logs",
        "Contact support"
    ]
}

# Recommended products

RECOMMENDED_PRODUCTS = {
    "installation": [
        "Installation Manager",
        "System Cleaner Tool"
    ],

    "network": [
        "Network Monitor",
        "VPN Software"
    ],

    "login": [
        "Password Manager",
        "Security Tool"
    ],

    "permission": [
        "User Access Manager",
        "Security Policy Tool"
    ],

    "update": [
        "Update Manager",
        "Driver Updater"
    ],

    "crash": [
        "System Optimizer",
        "Performance Monitor"
    ],

    "general": [
        "System Utility Tool",
        "Diagnostic Software"
    ]
}


# -----------------------------
# LOAD / SAVE STATS
# -----------------------------

def load_stats():

    if not os.path.exists(STATS_FILE):
        return {}

    with open(STATS_FILE, "r") as f:
        return json.load(f)


def save_stats(stats):

    with open(STATS_FILE, "w") as f:
        json.dump(stats, f)


def update_stats(category):

    stats = load_stats()

    if category not in stats:
        stats[category] = 0

    stats[category] += 1

    save_stats(stats)

    return stats


# -----------------------------
# CLEAN OCR TEXT
# -----------------------------

def clean_text(text):

    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    text = text.replace("\n\n", "\n")

    text = text.strip()

    return text


# -----------------------------
# CLASSIFY ERROR
# -----------------------------

def classify_error(text):

    text_lower = text.lower()

    for category, words in ERROR_KEYWORDS.items():

        for word in words:

            if word in text_lower:
                return category

    return "general"


# -----------------------------
# SEVERITY
# -----------------------------

def determine_severity(text):

    text_lower = text.lower()

    if "critical" in text_lower:
        return "Critical"

    if "failed" in text_lower or "error" in text_lower:
        return "High"

    if "warning" in text_lower:
        return "Medium"

    return "Low"


# -----------------------------
# FORMAT OUTPUT
# -----------------------------

def format_explanation(
    category,
    severity,
    solutions,
    products,
    stats,
    text
):

    stats_lines = ""

    for key, value in stats.items():
        stats_lines += f"{key}: {value}\n"

    explanation = f"""

Error Category:
{category.title()}

Severity Level:
{severity}

Suggested Solutions:
• {solutions[0]}
• {solutions[1] if len(solutions) > 1 else ""}

Recommended Products:
• {products[0]}
• {products[1] if len(products) > 1 else ""}

Error Statistics:
{stats_lines}

Detected Text:
{text}

"""

    return explanation


# -----------------------------
# MAIN FUNCTION
# -----------------------------

def analyze_uploaded_image(filepath):

    try:

        image = Image.open(filepath)

        text = pytesseract.image_to_string(image)

        text = clean_text(text)

        if not text:

            return (
                "No readable text detected",
                "The system could not detect any text in the screenshot.",
                0,
                ""
            )

        category = classify_error(text)

        severity = determine_severity(text)

        stats = update_stats(category)

        solutions = SOLUTIONS.get(
            category,
            SOLUTIONS["general"]
        )

        products = RECOMMENDED_PRODUCTS.get(
            category,
            RECOMMENDED_PRODUCTS["general"]
        )

        explanation = format_explanation(
            category,
            severity,
            solutions,
            products,
            stats,
            text
        )

        confidence = 90

        result = f"{category.title()} issue detected"

        return (

            result,
            explanation,
            confidence,
            text

        )

    except Exception as e:

        return (

            "Invalid image",
            f"Error analyzing image: {str(e)}",
            0,
            ""

        )