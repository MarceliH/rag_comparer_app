import os
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import chromadb
from chromadb.api import ClientAPI
import time


def get_chroma_collections(chroma_path: str | None = None):
    client = chromadb.PersistentClient(chroma_path if chroma_path else os.getenv("CHROMADB_PATH", ""))
    if not client:
        return []
    return client.list_collections()


def chroma_retriever(
    chroma_path: str | None,
    query_text: str,
    collection_name: str,
    n_results: int = 5,
):
    client = chromadb.PersistentClient(path=chroma_path if chroma_path else os.getenv("CHROMADB_PATH", ""))
    collection = client.get_collection(name=collection_name)
    embedding_function = DefaultEmbeddingFunction()
    query_embeddings = embedding_function([query_text]) # type: ignore

    start_time = time.time()
    retrieved_documents = collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results,
        include=["documents", "distances", "metadatas"]
    )
    retrieval_time = time.time() - start_time

    documents = retrieved_documents["documents"][0]
    document_names = [meta.get("source", "Unknown") for meta in retrieved_documents["metadatas"][0]]
    distances = retrieved_documents["distances"][0]

    return {
        "context": "\n\n".join(documents),
        "documents": documents,
        "document_names": document_names,
        "distances": distances,
        "retrieval_time": retrieval_time,
        "top_k": n_results
    }