# vector_store.py
import faiss
import numpy as np

dimension = 384  # embedding size for all-MiniLM-L6-v2
index = None
stored_chunks = []

def init_faiss(d=dimension):
    """
    Initialize the FAISS index and clear stored chunks.
    Returns the created index (also stored in global variable).
    """
    global index, stored_chunks, dimension
    dimension = d
    index = faiss.IndexFlatL2(dimension)
    stored_chunks = []
    return index

def add_to_index(embeddings, chunks):
    """
    Add embeddings and corresponding chunks to FAISS index.
    embeddings: iterable of vectors (list/ndarray) shape (n, dim)
    chunks: list of strings (same length as embeddings)
    """
    global stored_chunks, index
    if index is None:
        raise ValueError("FAISS index is not initialized. Call init_faiss() first.")
    emb_arr = np.array(embeddings).astype('float32')
    if emb_arr.ndim == 1:
        emb_arr = emb_arr.reshape(1, -1)
    if emb_arr.shape[0] != len(chunks):
        raise ValueError("Number of embeddings and chunks must match.")
    index.add(emb_arr)
    stored_chunks.extend(chunks)

def search_index(index_obj, query_or_embedding, top_k=3):
    """
    Search FAISS index for top K similar chunks.

    This function accepts:
      - index_obj: a faiss index instance (returned by init_faiss())
      - query_or_embedding: either a text string (will be embedded using chunk_embed.model)
                            or a numeric embedding (list/ndarray)
      - top_k: number of nearest neighbors to return

    Returns list of matching chunk strings (can be fewer than top_k if index is small).
    """
    global stored_chunks
    # Basic checks
    if index_obj is None:
        raise ValueError("Provided FAISS index is None. Make sure you passed init_faiss() result.")
    if not hasattr(index_obj, "search"):
        raise ValueError("First argument must be a FAISS index (object with .search method).")

    # If query is a string -> compute embedding using chunk_embed.model
    if isinstance(query_or_embedding, str):
        try:
            from chunk_embed import model
        except Exception as e:
            raise ImportError("Failed to import embedding model from chunk_embed.py") from e
        q_emb = model.encode([query_or_embedding])[0]
    else:
        # assume it's already an embedding vector / array-like
        q_emb = np.array(query_or_embedding).astype('float32')
        if q_emb.ndim == 2 and q_emb.shape[0] == 1:
            q_emb = q_emb.reshape(-1)
    # Prepare array for FAISS
    q_arr = np.array([q_emb]).astype('float32')

    # Run search
    D, I = index_obj.search(q_arr, top_k)

    results = []
    for idx in I[0]:
        if idx == -1:
            continue
        if 0 <= idx < len(stored_chunks):
            results.append(stored_chunks[idx])
    return results
