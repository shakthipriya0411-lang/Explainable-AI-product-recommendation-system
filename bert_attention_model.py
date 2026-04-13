import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# -----------------------------
# MODEL
# -----------------------------
class BertAttentionModel(nn.Module):

    def __init__(self):
        super(BertAttentionModel, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.attention = nn.Linear(768, 1)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        attention_scores = self.attention(hidden_states)
        attention_weights = torch.softmax(attention_scores, dim=1)

        context_vector = torch.sum(attention_weights * hidden_states, dim=1)
        logits = self.classifier(context_vector)

        return logits, attention_weights


# -----------------------------
# LOAD MODEL
# -----------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertAttentionModel()
model.eval()

# -----------------------------
# RULE-BASED HELP (IMPORTANT)
# -----------------------------
negative_words = ["bad", "damaged", "broken", "poor", "slow", "worst"]
positive_words = ["good", "excellent", "fast", "best", "great"]


# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict_sentiment(text):

    text_lower = text.lower()

    # ✅ Rule-based correction (VERY IMPORTANT FOR DEMO)
    if any(word in text_lower for word in negative_words):
        sentiment = "Negative"
    elif any(word in text_lower for word in positive_words):
        sentiment = "Positive"
    else:
        sentiment = "Neutral"

    # -----------------------------
    # BERT ATTENTION (for explanation)
    # -----------------------------
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    _, attention_weights = model(input_ids, attention_mask)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    attention_scores = attention_weights.squeeze().detach().numpy()

    word_attention = list(zip(tokens, attention_scores))

    word_attention = sorted(
        word_attention,
        key=lambda x: x[1],
        reverse=True
    )

    # Clean tokens
    important_words = [
        w[0].replace("##", "")
        for w in word_attention
        if w[0] not in ["[CLS]", "[SEP]"]
    ][:5]

    return sentiment, important_words