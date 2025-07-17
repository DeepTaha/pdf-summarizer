import streamlit as st
import tempfile
from rag_pipeline import *

st.set_page_config(page_title="RAG with Gemini", layout="wide")
st.title("📄 RAG-based PDF QA & Summarizer")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    st.success("✅ PDF uploaded!")
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    vectors, vectorizer = create_embeddings(chunks)
    index = create_faiss_index(vectors)

    # Summarization
    if st.button("🔍 Summarize PDF"):
        with st.spinner("Summarizing..."):
            summary = summarize_with_gemini(text)
            st.subheader("📌 Summary")
            st.write(summary)

    # Ask a question
    question = st.text_input("💬 Ask a question about the PDF:")
    if question:
        with st.spinner("Generating answer..."):
            context = query_faiss(question, chunks, index, vectorizer)
            answer = ask_gemini(question, context)
            st.subheader("💡 Answer")
            st.write(answer)


