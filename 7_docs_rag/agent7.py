import os
import pandas as pd
import dateparser
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from math_tools import (
    total_spent, average_spent, count_transactions,
    max_transaction, min_transaction
)

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Load CSV
df = pd.read_csv("final_transactions.csv")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.columns = df.columns.str.strip()

# Load FAISS index
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

# Load Groq LLM
llm = ChatGroq(api_key=groq_api_key, model_name="llama3-70b-8192")

# Prompt for RetrievalQA
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a smart financial assistant. Use the context below to answer the user's question about their credit card spending.

Context:
{context}

Question: {question}
Answer:
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=False
)

# Math-based logic
def handle_custom_query(query, df):
    q = query.lower()

    if "total spent" in q and "amazon" in q:
        return f"Total spent on Amazon: ₹{total_spent(df, merchant='Amazon'):.2f}"

    if "average" in q:
        return f"Average transaction amount: ₹{average_spent(df):.2f}"

    if "flipkart" in q and ("how many" in q or "number" in q):
        return f"Flipkart transactions: {count_transactions(df, merchant='Flipkart')}"

    if "maximum" in q or "highest" in q:
        txn = max_transaction(df)
        return f"Highest transaction: ₹{txn['Amount (INR)']} at {txn['Merchant']} on {txn['Date'].date()}"

    if "minimum" in q or "lowest" in q:
        txn = min_transaction(df)
        return f"Lowest transaction: ₹{txn['Amount (INR)']} at {txn['Merchant']} on {txn['Date'].date()}"

    if "total spent in" in q:
        try:
            month_text = q.split("total spent in")[1].strip()
            dt = dateparser.parse(month_text)
            if dt:
                month_df = df[df["Date"].dt.month == dt.month]
                total = month_df["Amount (INR)"].sum()
                return f"Total spent in {month_text.title()}: ₹{total:.2f}"
        except Exception:
            pass

    return None


# Main RAG loop
while True:
    question = input("\nAsk a question about your credit card spending (or type 'exit'): ")
    if question.lower() in ["exit", "quit"]:
        break

    try:
        custom = handle_custom_query(question, df)
        if custom:
            print("\n--- Answer ---")
            print(custom)
            continue

        # Otherwise use RAG with Groq + FAISS
        print("Searching relevant chunks...")
        response = qa_chain.run(question)
        print("\n--- Answer ---")
        print(response)

    except Exception as e:
        print("Error:", e)
