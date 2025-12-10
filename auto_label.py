import pandas as pd

df = pd.read_csv("cleaned_emails.csv")

def assign_label(text):

    text_lower = text.lower()

    if "urgent" in text_lower or "critical" in text_lower:
        return "Urgent"
    elif "payment" in text_lower or "invoice" in text_lower or "financial" in text_lower:
        return "Financial"
    elif "employee" in text_lower or "hr" in text_lower:
        return "HR"
    else:
        return "General"

df["category"] = df["text"].apply(assign_label)

df.to_csv("labeled_emails.csv", index=False)

print("Auto labeling completed!")
