import os
import google.generativeai as genai
from chunk_embed import model
from vector_store import search_index

# ✅ Configure Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

def answer_query(user_query, top_chunks):
    """
    Given a user query and retrieved text chunks, return a natural-language answer.
    """
    # 1️⃣ Embed the query
    query_embedding = model.encode([user_query])[0]

    # 2️⃣ Retrieve top 3 relevant chunks from FAISS
    context = top_chunks
    if not context:
        return "No relevant information found."

    # 3️⃣ Build the prompt
    prompt_text = (
        "You are a helpful assistant. Use only the following context to answer.\n\n"
        "Context:\n"
    )
    for idx, chunk in enumerate(context):
        prompt_text += f"[Chunk {idx+1}]: {chunk}\n\n"

    prompt_text += f"Question: {user_query}\nAnswer concisely:"

    # 4️⃣ Call Gemini 2.5 Flash Lite
    try:
        model_instance = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = model_instance.generate_content(prompt_text)

        if hasattr(response, "text"):
            return response.text.strip()
        elif hasattr(response, "candidates"):
            return response.candidates[0].content.parts[0].text.strip()
        else:
            return "No answer generated. Please verify the model output."
    except Exception as e:
        print("[ERROR] Gemini API call failed:", e)
        return f"Gemini error: {e}"
