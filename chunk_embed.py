# chunk_embed.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Initialize embedding model (small & fast)
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_text_chunks(text, chunk_size=1000, chunk_overlap=100):
    """
    Split text into overlapping chunks for better embedding search.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def generate_embeddings(chunks):
    """
    Convert text chunks to embeddings.
    """
    return model.encode(chunks)
