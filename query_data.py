import argparse
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()
FAISS_PATH = "faiss"

PROMPT_TEMPLATE = """Based on the following context, please answer the question. 
If you cannot find the exact answer, summarize what you know about the subject from the context.

Context:
{context}

Question: {question}

Answer:"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    try:
        print("Loading FAISS index...")
        index = FAISS.load_local(
            FAISS_PATH,
            embedding_function,
            allow_dangerous_deserialization=True
        )
        print(f"Number of documents in index: {index.index.ntotal}")

        # First try basic similarity search
        print("\nTrying basic similarity search first...")
        basic_results = index.similarity_search(query_text, k=8)
        print("\nBasic search results:")
        for i, doc in enumerate(basic_results, 1):
            print(f"\nResult {i}:")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")

        # Then try with relevance scores
        print("\nTrying similarity search with relevance scores...")
        results = index.similarity_search_with_relevance_scores(query_text, k=5)
        
        print("\nResults with scores:")
        for i, (doc, score) in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Score: {score:.4f}")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")

        # Lower threshold significantly for testing
        if len(results) == 0 or results[0][1] < 0.3:  # Lowered threshold to 0.3
            print(f"Unable to find relevant results above threshold.")
            return

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results[:3]])
        
        print("\nFinal context being used:")
        print(context_text[:500] + "...")  # Print first 500 chars of context

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        print("\nGenerating response...")
        model = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            model_kwargs={
                "temperature": 0.7,
                "max_length": 512,
                "top_p": 0.95,
                "do_sample": True
            },
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
        )

        response_text = model.invoke(prompt)

        print("\n=== Final Response ===")
        print(response_text)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()