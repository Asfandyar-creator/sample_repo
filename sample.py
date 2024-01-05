import streamlit as st
import fitz  # PyMuPDF
import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pypdf2
# Set your Gemini Pro API key
gemini_pro_api_key = "YOUR_GEMINI_PRO_API_KEY"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""

    for page_number in range(doc.page_count):
        page = doc[page_number]
        text += page.get_text()

    return text

# Function to get Gemini Pro embeddings
def get_gemini_pro_embeddings(texts):
    # Set up headers with the Gemini Pro API key
    headers = {
        "Authorization": f"Bearer {gemini_pro_api_key}"
    }

    # Request embeddings from Gemini Pro API
    response = requests.post("https://api.gemini.com/v1/embeddings", json={"texts": texts}, headers=headers)

    if response.status_code == 200:
        # Extract vector embeddings from the API response
        embeddings = response.json()["embeddings"]
        return embeddings
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None

# Function to find the most relevant passage based on cosine similarity
def find_most_relevant_passage(question, passages, embeddings):
    question_embedding = get_gemini_pro_embeddings([question])[0]
    passage_embeddings = get_gemini_pro_embeddings(passages)

    similarities = cosine_similarity([question_embedding], passage_embeddings)[0]
    most_relevant_index = np.argmax(similarities)

    return passages[most_relevant_index]

# Streamlit app
def main():
    st.title("PDF Reader with Gemini Pro API")

    # Upload PDF file
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if pdf_file is not None:
        # Process PDF file
        pdf_text = extract_text_from_pdf(pdf_file)

        # Get Gemini Pro embeddings for the entire document
        embeddings = get_gemini_pro_embeddings([pdf_text])

        if embeddings:
            # User input for questions
            question = st.text_input("Ask a question about the PDF:")

            if st.button("Answer"):
                # Find the most relevant passage based on cosine similarity
                relevant_passage = find_most_relevant_passage(question, [pdf_text], embeddings)
                st.subheader("Answer:")
                st.write(relevant_passage)
        else:
            st.warning("Failed to obtain embeddings. Please check your Gemini Pro API key.")

if __name__ == "__main__":
    main()
