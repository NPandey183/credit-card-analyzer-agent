import pandas as pd

# Load the cleaned data from anomaly detection output
df = pd.read_csv("../4_anomaly_detection/flagged_transactions.csv")

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# 📆 Monthly Spend Summary
monthly_summary = df.groupby(df['Date'].dt.to_period('M'))["Amount (INR)"].sum().reset_index()
monthly_summary.columns = ['Month', 'Amount (INR)']
monthly_summary['Month'] = monthly_summary['Month'].astype(str)

#  Category-wise Summary
category_summary = df.groupby("Category")["Amount (INR)"].sum().reset_index().sort_values(by="Amount (INR)", ascending=False)

#  Top 5 Merchants by Spend
top_merchants = df.groupby("Merchant")["Amount (INR)"].sum().reset_index().sort_values(by="Amount (INR)", ascending=False).head(5)

# Print all summaries
print("\n Monthly Summary:")
print(monthly_summary)

print("\n Category-wise Summary:")
print(category_summary)

print("\n Top 5 Merchants by Spend:")
print(top_merchants)

df.to_csv("final_transactions.csv", index=False)
