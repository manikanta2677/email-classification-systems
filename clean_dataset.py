import pandas as pd
import re

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    text = text.lower()
    return text

df = pd.read_csv("emails_5000.csv")

df["text"] = df["text"].astype(str).apply(clean_text)

df.drop_duplicates(subset=["text"], inplace=True)

df.to_csv("cleaned_emails.csv", index=False)

print("Dataset cleaned successfully!")
