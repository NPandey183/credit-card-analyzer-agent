# Credit Card Analyzer Agent

This project is an AI-powered assistant for analyzing credit card transactions. Users can upload their transaction CSV files and ask questions in natural language, such as "How much did I spend on Amazon in June?" The system provides meaningful responses by combining data processing, categorization, summary generation, and large language models.

The solution is built using LangChain, Groq LLM, FAISS for vector retrieval, and Streamlit for the user interface.

---

## Features

- Upload and process credit card transaction CSV files
- Ask natural language questions about your expenses
- Detect anomalies and outliers in transactions
- Generate summaries by month and category
- Retrieve semantically relevant chunks using a document-based RAG system
- Answer complex queries using a Groq-powered LangChain agent

---

## Project Structure

credit-card-analyzer-agent/
├── 1_data/ # Sample credit card CSVs
├── 2_preprocessing/ # Data cleaning and formatting scripts
├── 3_categorization/ # Predict and label expense categories
├── 4_anomaly_detection/ # Outlier detection logic
├── 5_summary_generation/ # Generate monthly/category-wise summaries
├── 6_qa_agent_groq/ # Groq LLM agent with LangChain
├── 7_docs_rag/ # Document-based RAG pipeline
├── 8_streamlit_ui/ # Streamlit frontend application
├── requirements.txt # Project dependencies
└── .gitignore # Ignored files and folders

## Technologies Used

- LangChain for agent creation and orchestration
- Groq LLM for fast and context-aware responses
- Pandas and NumPy for data manipulation
- FAISS for semantic search and retrieval
- Streamlit for the user interface
- dotenv for managing API keys

## License

This project is for educational and demonstration purposes only.

## Limitations
This project is still a work in progress. While it demonstrates key capabilities of AI-powered transaction analysis, there are several known limitations:

The system currently relies on a specific CSV structure and may require manual adjustments for other formats.

Query handling is based on Groq LLM and may occasionally produce inaccurate or incomplete responses.

Category prediction and anomaly detection use basic logic and can be improved with more advanced models.


Some parts of the RAG pipeline and fallback logic are experimental and may be refined in future updates.

