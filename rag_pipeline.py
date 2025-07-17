import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import numpy as np

# 🌐 Configure Gemini API from Streamlit Secrets
genai.configure(api_key=st.secrets["AIzaSyDWypnP5Ds8_Yw3myn1yzDFxswasR8aBWM"])
model = genai.GenerativeModel("gemini-2.5-flash")

# 📄 Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# 📚 Chunk text into overlapping pieces
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# 📐 Create TF-IDF embeddings for text chunks
def create_embeddings(chunks):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(chunks).toarray()
    return vectors, vectorizer

# 🧠 Store vectors in FAISS index
def create_faiss_index(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype('float32'))
    return index

# 🔍 Retrieve relevant chunks from FAISS for a question
def query_faiss(query, chunks, index, vectorizer, k=3):
    query_vec = vectorizer.transform([query]).toarray().astype('float32')
    D, I = index.search(query_vec, k)
    return "\n".join([chunks[i] for i in I[0]])

# 💬 Ask Gemini a question using retrieved context
def ask_gemini(question, context):
    prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {question}"
    response = model.generate_content(prompt)
    return response.text

# 📝 Summarize the whole PDF using Gemini
def summarize_with_gemini(text):
    prompt = f"Summarize the following research paper:\n\n{text}"
    response = model.generate_content(prompt)
    return response.text
