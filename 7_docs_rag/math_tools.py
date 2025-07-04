import pandas as pd

def total_spent(df, merchant=None, category=None):
    temp = df.copy()
    if merchant:
        temp = temp[temp["Merchant"].str.lower() == merchant.lower()]
    if category:
        temp = temp[temp["Category"].str.lower() == category.lower()]
    return temp["Amount (INR)"].sum()

def average_spent(df, merchant=None):
    temp = df.copy()
    if merchant:
        temp = temp[temp["Merchant"].str.lower() == merchant.lower()]
    return temp["Amount (INR)"].mean()

def count_transactions(df, merchant=None):
    temp = df.copy()
    if merchant:
        temp = temp[temp["Merchant"].str.lower() == merchant.lower()]
    return temp.shape[0]

def max_transaction(df):
    return df.loc[df["Amount (INR)"].idxmax()]

def min_transaction(df):
    return df.loc[df["Amount (INR)"].idxmin()]
