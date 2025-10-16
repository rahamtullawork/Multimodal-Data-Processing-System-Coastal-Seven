import streamlit as st
from pathlib import Path
import tempfile
import os

# Import helper modules
from chunk_embed import model
from file_parser import extract_text_from_file
from image_ocr import extract_text_from_image
from av_transcriber import transcribe_media
from chunk_embed import get_text_chunks, generate_embeddings
from vector_store import init_faiss, add_to_index, search_index
from query_handler import answer_query
import utils


# -------------------------------
# Streamlit UI setup
# -------------------------------
st.set_page_config(page_title="Multimodal Data Processing System", layout="centered")
st.title("🤖 Multimodal Data Processing System")
st.write("Upload any text, image, or audio/video file and ask natural language questions!")

# Initialize FAISS index (simple, in-memory)
index = init_faiss()

# Temporary directory for uploaded files
temp_dir = tempfile.gettempdir()


# -------------------------------
# File Upload Section
# -------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload File (PDF, DOCX, PPTX, TXT, MD, PNG, JPG, MP3, MP4)",
    type=["pdf", "docx", "pptx", "txt", "md", "png", "jpg", "jpeg", "mp3", "mp4"]
)

if uploaded_file is not None:
    file_path = Path(temp_dir) / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ Uploaded: {uploaded_file.name}")

    file_type = uploaded_file.type

    # -------------------------------
    # Extract text depending on file type
    # -------------------------------
    text = ""
    if any(file_path.suffix.lower() in ext for ext in [".pdf", ".docx", ".pptx", ".txt", ".md"]):
        text = extract_text_from_file(file_path)
    elif any(file_path.suffix.lower() in ext for ext in [".png", ".jpg", ".jpeg"]):
        text = extract_text_from_image(file_path)
    elif any(file_path.suffix.lower() in ext for ext in [".mp3", ".mp4"]):
        text = transcribe_media(file_path)
    else:
        st.error("Unsupported file format!")
        st.stop()

    if not text.strip():
        st.warning("No readable text found in file.")
        st.stop()

    st.write("📄 **Extracted Text (first 10000 chars):**")
    st.text(text[:10000] + "..." if len(text) > 10000 else text)

    # -------------------------------
    # Chunking & Embeddings
    # -------------------------------
    chunks = get_text_chunks(text)
    embeddings = generate_embeddings(chunks)

    add_to_index(embeddings, chunks)
    st.success(f"✅ Indexed {len(chunks)} text chunks from {uploaded_file.name}")


# -------------------------------
# Query Section
# -------------------------------
st.subheader("💬 Ask a Question")
query = st.text_input("Enter your natural language query here")

if st.button("Search and Answer"):
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        context_chunks = search_index(index, query)
        response = answer_query(query, context_chunks)
        st.write("### 🤖 Answer:")
        st.write(response)