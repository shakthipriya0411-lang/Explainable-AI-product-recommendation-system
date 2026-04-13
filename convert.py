import json
import pandas as pd

print("Converting JSON to CSV...")

data = []

with open("Software_5.json", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 10000:   # Limit to 10,000 reviews
            break
        
        review = json.loads(line)
        
        if 'reviewText' in review and 'overall' in review:
            data.append({
                "user_id": review.get("reviewerID"),
                "product_id": review.get("asin"),
                "rating": review.get("overall"),
                "review_text": review.get("reviewText")
            })

df = pd.DataFrame(data)

df.to_csv("amazon_software.csv", index=False)

print("CSV Created Successfully!")
print("Total reviews extracted:", len(df))
