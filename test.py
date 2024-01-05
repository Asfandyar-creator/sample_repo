import streamlit as st
#import gemini
import pypdf2
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load Gemini model and tokenizer
model_name = "google/gemini-replica-137B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Define functions for PDF reading and question answering
def read_pdf(pdf_file):
    # Extract text from PDF using PyPDF2
    with open(pdf_file, "rb") as pdf:
        pdf_reader = pypdf2.PdfReader(pdf)
        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
    return pdf_text

def answer_question(question, pdf_text):
    # Format prompt for Gemini using PDF text and question
    prompt = f"I have read a PDF document containing the following information:\n{pdf_text}\nNow, I'll try to answer your question:\n{question}"

    # Generate response using Gemini
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Streamlit layout
st.title("PDF Reader with Gemini")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    pdf_text = read_pdf(uploaded_file)

    # Question input
    question = st.text_input("Ask a question about the PDF")

    if question:
        response = answer_question(question, pdf_text)
        st.write(response)
