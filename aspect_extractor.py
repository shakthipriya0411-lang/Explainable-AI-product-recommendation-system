import spacy

nlp = spacy.load("en_core_web_sm")

def extract_aspects(text):

    doc = nlp(text)

    aspects = []

    for token in doc:
        if token.pos_ == "NOUN":
            aspects.append(token.text)

    return list(set(aspects))