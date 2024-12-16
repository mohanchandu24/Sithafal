from utils.pdf_extraction import extract_text_from_pdf
from utils.text_chunking import chunk_text
from utils.embeddings import create_embeddings, query_pdf
from utils.response_generation import generate_response

def main():
    # Step 1: Extract text from the PDF
    pdf_path = "sample.pdf"  # Replace with your PDF
    print("Extracting text from PDF...")
    pdf_text = extract_text_from_pdf(pdf_path)
    
    # Step 2: Chunk the text
    print("Chunking text into smaller pieces...")
    chunks = chunk_text(pdf_text)

    # Step 3: Generate embeddings and create a FAISS index
    print("Creating embeddings and building index...")
    index, embeddings, model = create_embeddings(chunks)

    # Step 4: Query the system
    user_query = "What is the unemployment rate for bachelor's degrees?"
    print(f"Querying: {user_query}")
    retrieved_chunks = query_pdf(index, user_query, model, chunks)

    # Step 5: Generate a response
    final_response = generate_response(user_query, retrieved_chunks)
    print("Final Response:\n", final_response)

if __name__ == "__main__":
    main()
