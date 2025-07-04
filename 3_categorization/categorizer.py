import pandas as pd

# Load cleaned data
df = pd.read_csv("../2_preprocessing/cleaned_transactions.csv")

# Rule-based categorization (can be improved with ML later)
merchant_category_map = {
    "Big Bazaar": "Grocery",
    "Uber": "Travel",
    "Netflix": "Entertainment",
    "Indian Oil": "Fuel",
    "Hotstar": "Entertainment",
    "Airtel": "Bills",
    "Zomato": "Food",
    "Amazon": "Shopping",
    "Flipkart": "Shopping",
    "Ola": "Travel",
    "Spencer’s": "Grocery"
}

# Apply mapping
df["Predicted Category"] = df["Merchant"].map(merchant_category_map).fillna("Other")

print(df[["Merchant", "Predicted Category"]].head(10))

# Save to CSV
df.to_csv("categorized_transactions.csv", index=False)
