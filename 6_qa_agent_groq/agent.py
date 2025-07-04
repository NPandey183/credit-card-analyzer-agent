from math_tools import total_spent, average_spent, count_transactions, max_transaction, min_transaction

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Load the Groq LLM (LLaMA 3)
llm = ChatGroq(api_key=groq_api_key, model_name="llama3-70b-8192")

# Load transaction data
df = pd.read_csv("final_transactions.csv")
df.columns = df.columns.str.strip()




# Ensure 'Date' is parsed correctly
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df[df["Date"].notna()]  # Drop rows with invalid dates

# Create the Pandas DataFrame agent
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    allow_dangerous_code=True
)
agent.handle_parsing_errors = True  # Gracefully skip bad LLM output

# Start interactive QA loop
def handle_custom_math_query(query, df):
    q = query.lower()

    if "total spent" in q and "amazon" in q:
        return f"Total spent on Amazon: ₹{total_spent(df, merchant='Amazon'):.2f}"

    if "average" in q:
        return f"Average transaction amount: ₹{average_spent(df):.2f}"

    if "how many" in q or "number of" in q:
        if "flipkart" in q:
            return f"Number of Flipkart transactions: {count_transactions(df, merchant='Flipkart')}"
        else:
            return f"Total number of transactions: {df.shape[0]}"

    if "highest" in q or "maximum" in q:
        txn = max_transaction(df)
        return f"Highest transaction: ₹{txn['Amount']} at {txn['Merchant']} on {txn['Date'].date()}"

    if "lowest" in q or "minimum" in q:
        txn = min_transaction(df)
        return f"Lowest transaction: ₹{txn['Amount']} at {txn['Merchant']} on {txn['Date'].date()}"

    return None


while True:
    question = input("\nAsk your question (or type 'exit'): ")
    if question.lower() in ["exit", "quit"]:
        break
    try:
        custom_answer = handle_custom_math_query(question, df)
        if custom_answer:
            print("\n--- Answer ---")
            print(custom_answer)
            continue

        modified_question = question.strip()
        if not modified_question.lower().startswith("print"):
            modified_question += " — and print the final result."

        response = agent.invoke({"input": modified_question})
        print("\n--- Answer ---")
        print(response["output"])
    except Exception as e:
        print("\nError occurred:", str(e))



