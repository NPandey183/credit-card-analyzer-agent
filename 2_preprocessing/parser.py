import pandas as pd

# Read the CSV from the data folder
df = pd.read_csv("../1_data/sample_credit_card_transactions.csv")

# Format date
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Save cleaned version
df.to_csv("cleaned_transactions.csv", index=False)

print(df.head())
