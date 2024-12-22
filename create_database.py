# Các import cho việc load và xử lý tài liệu
from langchain_community.document_loaders import DirectoryLoader  # Để load tài liệu từ thư mục
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Để chia nhỏ văn bản
from langchain.schema import Document  # Schema cho tài liệu
from langchain_openai import OpenAIEmbeddings  # Để tạo embeddings sử dụng OpenAI
from langchain_community.vectorstores import FAISS    # Vector store để lưu trữ
import openai  # OpenAI API
from dotenv import load_dotenv  # Để load biến môi trường
import os, shutil  # Các thao tác với hệ thống

load_dotenv()  # Load biến môi trường từ file .env
openai.api_key = os.environ['OPENAI_API_KEY']  # Thiết lập API key

# Định nghĩa các đường dẫn
FAISS_PATH = "faiss"  # Thư mục lưu database vector
DATA_PATH = "data/books"  # Thư mục chứa dữ liệu đầu vào


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_faiss(chunks)


def load_documents():
    # Load tất cả file .md trong thư mục DATA_PATH
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents):
    # Tạo text splitter để chia nhỏ văn bản
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Độ dài mỗi chunk
        chunk_overlap=100,  # Độ chồng lấp giữa các chunk
        length_function=len,
        add_start_index=True,
    )
    # Chia nhỏ tài liệu
    chunks = text_splitter.split_documents(documents)
    
    # In thông tin debug
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    document = chunks[10]  # Xem ví dụ chunk thứ 10
    print(document.page_content)
    print(document.metadata)
    
    return chunks


def save_to_faiss(chunks):
    # Xóa database cũ nếu tồn tại
    if os.path.exists(FAISS_PATH):
        shutil.rmtree(FAISS_PATH)
    
    # Tạo database mới từ các chunks
    db = FAISS.from_documents(
        chunks,  # Các đoạn văn bản đã chia nhỏ
        OpenAIEmbeddings(),  # Sử dụng OpenAI để tạo embeddings
        persist_directory=FAISS_PATH  # Lưu vào thư mục
    )
    db.persist()  # Lưu database
    print(f"Saved {len(chunks)} chunks to {FAISS_PATH}.")


if __name__ == "__main__":
    main()