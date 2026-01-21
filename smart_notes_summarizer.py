# smart_notes_summarizer.py

import streamlit as st
from transformers import pipeline
#import PyPDF2
from pypdf import PdfReader

# -------------------------
# Helper Functions
# -------------------------

def extract_pdf_text(pdf_file):
    """Extract text from uploaded PDF."""
    text = ""
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def summarize_text(text, max_length=150, min_length=50):
    """Summarize text using Hugging Face Transformers."""
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # Transformers models have a token limit; split if too long
    # For simplicity, split into chunks of ~1000 words
    import math
    words = text.split()
    chunk_size = 1000
    chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
    
    summaries = []
    for chunk in chunks:
        chunk_text = " ".join(chunk)
        summary = summarizer(chunk_text, max_length=max_length, min_length=min_length, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    
    return " ".join(summaries)

# -------------------------
# Streamlit App
# -------------------------

st.set_page_config(page_title="Smart Notes Summarizer", page_icon="📝")

st.title("📝 Smart Notes Summarizer")
st.write("Upload your notes (PDF or TXT), and get an exam-ready summary instantly!")

# File uploader
uploaded_file = st.file_uploader("Upload your notes", type=["pdf","txt"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        text = extract_pdf_text(uploaded_file)
    elif uploaded_file.type == "text/plain":
        text = str(uploaded_file.read(), "utf-8")
    else:
        st.error("Unsupported file type!")
        text = None

    if text:
        st.subheader("Original Text Preview:")
        st.text_area("Text Preview", text[:1000] + ("..." if len(text) > 1000 else ""), height=200)
        
        if st.button("Generate Summary"):
            with st.spinner("Summarizing... ⏳"):
                summary = summarize_text(text)
            st.subheader("📄 Summary:")
            st.write(summary)
