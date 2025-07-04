from dotenv import load_dotenv
load_dotenv()

import os
import pickle
import faiss
import re
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage

import pandas as pd
import sys

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Load the credit card dataset
df = pd.read_csv("final_transactions.csv")
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
df['Amount (INR)'] = df['Amount (INR)'].replace('[₹,]', '', regex=True).astype(float)

# Load FAISS index
print("Loading FAISS index and chunks...")
with open("chunks.pkl", "rb") as f:
    texts = pickle.load(f)

faiss_index = FAISS.load_local(
    "faiss_index",
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True
)

# Load LLM
print("Loading embedding model for query...")
llm = ChatGroq(temperature=0, model_name="LLaMA3-8b-8192")

# Create retriever chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=faiss_index.as_retriever(),
    return_source_documents=True
)

# === Hardcoded logic ===

def get_top_spends(df, top_n=5):
    top_df = df.sort_values(by='Amount (INR)', ascending=False).head(top_n)
    return top_df[['Date', 'Merchant', 'Amount (INR)', 'Category']]

def handle_special_query(query: str):
    query = query.lower()

    if "top" in query and "spend" in query:
        print("\nTop spending transactions:")
        top_df = get_top_spends(df, top_n=10 if "10" in query else 5)
        for idx, row in top_df.iterrows():
            print(f"{row['Date']} – {row['Merchant']} – ₹{row['Amount (INR)']:.2f} – {row['Category']}")
        return True

    elif "amazon" in query and "spend" in query:
        amazon_txns = df[df["Merchant"].str.contains("amazon", case=False)]
        if not amazon_txns.empty:
            print("\nAmazon transactions:")
            for idx, row in amazon_txns.iterrows():
                print(f"{row['Date']} – ₹{row['Amount (INR)']:.2f} – {row['Category']}")
        else:
            print("No Amazon transactions found.")
        return True

    return False


# === Main loop ===
while True:
    query = input("\nAsk a question about your credit card spending (or type 'exit'): ")
    if query.lower() == "exit":
        break

    # Handle custom logic first
    if handle_special_query(query):
        continue

    # Else, fall back to vector search + LLM
    print("Searching relevant chunks...")
    response = qa_chain.invoke({"query": query})
print("\nAnswer:\n", response["result"])
