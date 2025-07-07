def search_drive_documents(query: str, top_k: int = 5) -> list:
    """
    Tool: Search the embedded Google Drive documents using a natural language query.

    Args:
        query (str): The user's question or search phrase.
        top_k (int): Number of top matching document chunks to return (default = 5).

    Returns:
        List[dict]: A list of dictionaries. Each dictionary contains:
            - 'text': The matching chunk of document text
            - 'file_name': The name of the file it came from
            - 'file_id': The Google Drive file ID
    """
    from sentence_transformers import SentenceTransformer
    from chromadb import PersistentClient

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    chroma_client = PersistentClient(path="./chroma_store")
    collection = chroma_client.get_or_create_collection(name="drive-docs")

    query_embedding = embedding_model.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    matches = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        matches.append({
            "text": doc,
            "file_name": meta.get("file_name", "unknown"),
            "file_id": meta.get("file_id", "unknown")
        })

    return matches