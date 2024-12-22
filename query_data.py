import argparse
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
FAISS_PATH = "faiss"

PROMPT_TEMPLATE = """
Answer the question based only on the following context. If the answer cannot be found in the context, say "I don't know based on the given context."

Context:
{context}

Question: {question}
Answer: Let me answer based on the context provided."""

def main():
    # Create CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Initialize embeddings
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Check if FAISS index exists
    if not os.path.exists(FAISS_PATH):
        print(f"Error: FAISS index not found at {FAISS_PATH}")
        return

    try:
        # Load the existing FAISS index
        print("Loading FAISS index...")
        index = FAISS.load_local(
            FAISS_PATH,
            embedding_function,
            allow_dangerous_deserialization=True
        )
        print("FAISS index loaded successfully")

        # Search the DB
        print(f"Searching for: {query_text}")
        results = index.similarity_search_with_relevance_scores(query_text, k=3)
        
        if len(results) == 0 or results[0][1] < 0.7:  # Increased threshold
            print(f"Unable to find relevant results.")
            return

        # Prepare context from results
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        # Create prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        # Initialize HuggingFace model
        print("Generating response...")
        model = HuggingFaceHub(
            repo_id="google/flan-t5-large",  # Using larger model
            model_kwargs={
                "temperature": 0.3,  # Reduced temperature for more focused answers
                "max_length": 512,
                "top_p": 0.9
            },
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
        )

        # Generate response
        response_text = model.invoke(prompt)  # Updated to use invoke instead of predict

        # Get sources with scores
        sources_with_scores = [(doc.metadata.get("source", None), score) 
                             for doc, score in results]

        # Format and print response
        print("\n=== Response ===")
        print(response_text)
        print("\n=== Sources and Relevance Scores ===")
        for source, score in sources_with_scores:
            print(f"Source: {source}")
            print(f"Relevance Score: {score:.4f}\n")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()