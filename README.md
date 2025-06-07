# LangChain RAG System

A simple Retrieval-Augmented Generation (RAG) system using LangChain, FAISS, and HuggingFace to search and answer questions based on documents.

## Features

* **Document Processing:** Reads and processes text documents from Markdown files.
* **Vector Embeddings:** Generates vector embeddings using HuggingFace models.
* **Vector Storage and Search:** Stores and searches vectors using FAISS.
* **Contextual Q\&A:** Answers questions based on context retrieved from documents.

## System Requirements

* Python 3.8+
* pip (Python package manager)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/trngthnh369/langchain-rag-LLMs.git
   cd langchain-rag-LLMs
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Create the Vector Database

Process the documents and create a FAISS vector database.

```bash
python create_database.py
```

This script will:

* Read documents from the `data/` directory.
* Generate embeddings using HuggingFace models.
* Store the embeddings in a FAISS index.

### 2. Query the Data

Ask questions based on the processed documents.

```bash
python query_data.py
```

You'll be prompted to enter a question. The system will:

* Retrieve relevant documents from the FAISS index.
* Use LangChain to generate an answer based on the retrieved context.

### 3. Compare Embeddings (Optional)

If you wish to compare different embedding models:

```bash
python compare_embeddings.py
```

This script allows you to evaluate and compare the performance of various embedding models on your dataset.

## Customization

* **Adding Documents:** Place your Markdown (`.md`) files in the `data/` directory. Rerun `create_database.py` to process the new documents.
* **Changing Embedding Models:** Modify the embedding model used in `create_database.py` and `query_data.py` by selecting a different HuggingFace model.
* **Adjusting FAISS Parameters:** Tweak the FAISS index parameters in `create_database.py` to optimize search performance.

## Project Structure

```
langchain-rag-LLMs/
├── data/
│   └── books/                 # Directory containing Markdown documents
├── create_database.py         # Script to create FAISS vector database
├── query_data.py              # Script to query the vector database
├── compare_embeddings.py      # Script to compare different embedding models
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Acknowledgements

* [LangChain](https://github.com/hwchase17/langchain) for providing the framework to build LLM-powered applications.
* [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search and clustering of dense vectors.
* [HuggingFace](https://huggingface.co/) for access to a wide range of pre-trained models.

## Contact

For questions or support, please contact: truongthinhnguyen30303@gmail.com