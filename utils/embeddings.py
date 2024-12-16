from sentence_transformers import SentenceTransformer
import faiss

def create_embeddings(chunks):
    """Generates embeddings for text chunks and builds a FAISS index."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_numpy=True)
    
    # Store embeddings in a FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings, model

def query_pdf(index, query, model, chunks, top_k=3):
    """Handles user queries and retrieves the most relevant chunks."""
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = [chunks[i] for i in indices[0]]
    return results
