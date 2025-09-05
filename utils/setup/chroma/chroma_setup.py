import chromadb

def check_chroma_connection(chroma_path: str):
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        # Try to list collections to verify connection
        _ = client.list_collections()
        return True
    except Exception:
        return False