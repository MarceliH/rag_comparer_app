from http import client
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import chromadb
from chromadb.api import ClientAPI
import time
import os
import logging

def instantiate_chroma_client(chroma_path: str | None = None) -> ClientAPI | None:
    try:
        if chroma_path is not None:
            return chromadb.PersistentClient(path=chroma_path)
        env_path = os.getenv("CHROMADB_PATH")
        if env_path is None:
            return None
        return chromadb.PersistentClient(path=env_path)
    except (OSError, ValueError) as e:
        logging.error(f"Error instantiating ChromaDB client: {e}")
        return None


def get_chroma_collections(chroma_client: ClientAPI) -> list[str]:
    if not chroma_client:
        return []
    return [str(col) for col in chroma_client.list_collections()]


def chroma_retriever(
    chroma_client: ClientAPI,
    query_text: str,
    collection_name: str,
    n_results: int = 5,
):
    collection = chroma_client.get_collection(name=collection_name)
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