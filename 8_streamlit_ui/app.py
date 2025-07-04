import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os

# LangChain imports
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Streamlit page setup
st.set_page_config(page_title="Credit Card Statement Analyzer", layout="centered")
st.title("Credit Card Statement Analyzer")

st.write("Upload your credit card CSV file, then ask questions like:")
st.markdown("""
- Total spent on Amazon  
- Average spend in June  
- Highest transaction in May  
""")

# Upload section
uploaded_file = st.file_uploader("Upload your credit card statement (.csv)", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.lower()
        st.success("File uploaded successfully.")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    # Step 1: Smart column matching
    def smart_match(columns, keywords):
        for col in columns:
            for keyword in keywords:
                if keyword in col.lower():
                    return col
        return None

    columns = df.columns

    date_col = smart_match(columns, ["date", "transaction date", "txn date"])
    amount_col = smart_match(columns, ["amount", "amt", "transaction amount", "value"])
    merchant_col = smart_match(columns, ["merchant", "vendor", "shop", "biller", "company"])
    category_col = smart_match(columns, ["category", "type"])
    mode_col = smart_match(columns, ["mode", "payment", "method"])
    predicted_col = smart_match(columns, ["predicted", "label", "tag"])

    # Step 2: Validate required columns
    if not all([date_col, amount_col, merchant_col]):
        st.error("Required columns like amount, date, or merchant are missing or couldn't be inferred.")
        st.stop()

    # Step 3: Parse date
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")

    # Step 4: Prepare text rows
    rows = []
    for _, row in df.iterrows():
        try:
            content = (
                f"Date: {row.get(date_col, 'N/A')}\n"
                f"Merchant: {row.get(merchant_col, 'N/A')}\n"
                f"Amount (INR): ₹{row.get(amount_col, 'N/A')}\n"
                f"Category: {row.get(category_col, 'N/A')}\n"
                f"Mode: {row.get(mode_col, 'N/A')}\n"
                f"Predicted Category: {row.get(predicted_col, 'N/A')}"
            )
            rows.append(content)
        except:
            continue

    if not rows:
        st.error("No valid rows found in the CSV.")
        st.stop()

    # Step 5: Embedding model and FAISS
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    documents = [Document(page_content=row) for row in rows]
    vector_store = FAISS.from_documents(documents, embedding_model)

    # Step 6: Ask questions
    query = st.text_input("Ask a question about your credit card statement:")

    if query:
        similar_docs = vector_store.similarity_search(query, k=5)

        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-70b-8192"
        )

        qa_chain = load_qa_chain(llm, chain_type="stuff")
        answer = qa_chain.run(input_documents=similar_docs, question=query)

        st.markdown("## Answer")
        st.success(answer)
