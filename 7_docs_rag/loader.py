import pandas as pd

def load_and_chunk_csv(csv_path: str):
    # Load the CSV
    df = pd.read_csv(csv_path)

    # Convert Date column to datetime (DD-MM-YYYY style)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df[df["Date"].notna()]  # Remove rows with invalid dates

    # Fill missing optional columns with default values
    for col in ['Category', 'Description', 'Mode', 'Predicted Category']:
        if col not in df.columns:
            df[col] = 'N/A'

    # Generate structured text chunks
    chunks = []
    for _, row in df.iterrows():
        chunk = f"""
Date: {row['Date'].strftime('%Y-%m-%d')}
Merchant: {row.get('Merchant', 'N/A')}
Amount (INR): {row['Amount (INR)']}
Category: {row['Category']}
Mode: {row['Mode']}
Predicted Category: {row['Predicted Category']}
Description: {row['Description']}
""".strip()
        chunks.append(chunk)

    return chunks

# Run standalone to preview chunks
if __name__ == "__main__":
    chunks = load_and_chunk_csv("../5_summary_generation/final_transactions.csv")
    print(f"\n Loaded {len(chunks)} chunks. Sample preview:\n")
    for line in chunks[:3]:
        print(line + "\n" + "-"*40)

