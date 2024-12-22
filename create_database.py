import glob
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

CHUNK_SIZE = 300
CHUNK_OVERLAP = 100
DATA_PATH = os.path.join("data", "books", "*.md")  # Sửa lại đường dẫn
FAISS_PATH = "faiss"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    if not documents:
        print("No documents found!")
        return
    chunks = split_text(documents)
    save_to_faiss(chunks)

def load_documents():
    all_documents = []
    for file_path in glob.glob(DATA_PATH):
        try:
            # Chuyển đổi đường dẫn sang absolute path
            abs_file_path = os.path.abspath(file_path)
            print(f"Loading file: {abs_file_path}")
            
            # Kiểm tra file có tồn tại không
            if not os.path.exists(abs_file_path):
                print(f"File not found: {abs_file_path}")
                continue
                
            loader = TextLoader(abs_file_path, encoding='utf-8')  # Thêm encoding
            documents = loader.load()
            all_documents.extend(documents)
            print(f"Successfully loaded: {abs_file_path}")
        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
    return all_documents

def split_text(documents: List):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Print a sample chunk
    if len(chunks) > 0:
        print(chunks[0].page_content)
        print(chunks[0].metadata)
    
    return chunks

def save_to_faiss(chunks: List):
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create and save FAISS index
    db = FAISS.from_documents(chunks, embeddings)
    
    # Ensure the directory exists
    os.makedirs(FAISS_PATH, exist_ok=True)
    
    # Save the index
    db.save_local(FAISS_PATH)

if __name__ == "__main__":
    main()