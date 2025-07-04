import pandas as pd

# Load data with categories
df = pd.read_csv("../3_categorization/categorized_transactions.csv")

# Convert date and amount
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Amount (INR)"] = pd.to_numeric(df["Amount (INR)"], errors="coerce")

# Anomaly rules
df["High Value Flag"] = df["Amount (INR)"] > 5000
df["Duplicate Flag"] = df.duplicated(subset=["Date", "Amount (INR)", "Merchant"], keep=False)
suspicious_merchants = ["Unknown", "Xyz Corp", "Test Merchant"]
df["Suspicious Merchant Flag"] = df["Merchant"].isin(suspicious_merchants)

# Save anomalies
df.to_csv("analyzed_transactions.csv", index=False)

# Show sample
print(df[["Date", "Merchant", "Amount (INR)", "High Value Flag", "Duplicate Flag", "Suspicious Merchant Flag"]].head(15))

df.to_csv("flagged_transactions.csv", index=False)
