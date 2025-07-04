# embedder.py

from sentence_transformers import SentenceTransformer
import faiss
import pickle
from loader import load_and_chunk_csv

def embed_chunks(chunks, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings

def save_faiss_index(embeddings, chunks, index_path='credit_card_index.faiss', metadata_path='chunks.pkl'):
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, index_path)

    # Save metadata
    with open(metadata_path, 'wb') as f:
        pickle.dump(chunks, f)

    print(f" FAISS index saved to {index_path}")
    print(f" Chunk metadata saved to {metadata_path}")

if __name__ == "__main__":
    print(" Loading and chunking CSV...")
    chunks = load_and_chunk_csv("../5_summary_generation/final_transactions.csv")

    print(f" Embedding {len(chunks)} chunks...")
    embeddings = embed_chunks(chunks)

    print(" Saving FAISS index and chunk metadata...")
    save_faiss_index(embeddings, chunks)
