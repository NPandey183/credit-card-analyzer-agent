import pandas as pd
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load the CSV
df = pd.read_csv("final_transactions.csv")
df.columns = df.columns.str.strip()
df['Amount (INR)'] = df['Amount (INR)'].replace('[₹,]', '', regex=True).astype(float)

# Combine rows into a single string format
text_rows = df.apply(
    lambda row: f"{row['Date']} - {row['Merchant']} - ₹{row['Amount (INR)']} - {row['Category']}",
    axis=1
).tolist()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.create_documents(text_rows)

# Save chunks
with open("chunks.pkl", "wb") as f:
    pickle.dump([doc.page_content for doc in chunks], f)

# Create FAISS index
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
faiss_index = FAISS.from_documents(chunks, embedding_model)

# Save index
faiss_index.save_local("faiss_index")

print(" chunks.pkl and faiss_index saved successfully.")
