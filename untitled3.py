import streamlit as st
import os
import pandas as pd

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq

# Page Config
st.set_page_config(page_title="PragyanAI Assistant", page_icon="🤖")

st.title("🤖 PragyanAI Intelligent Assistant")

# Load API Key from Streamlit Secrets
groq_api_key = st.secrets["GROQ_API_KEY"]

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAQ Excel
@st.cache_resource
def load_vectorstore():
    docs = []

    if os.path.exists("pragyan_faq_prices.xlsx"):
        df = pd.read_excel("pragyan_faq_prices.xlsx")

        for _, row in df.iterrows():
            content = " | ".join(
                [f"{col}: {val}" for col, val in row.items()]
            )
            docs.append(
                Document(page_content=content)
            )

    return FAISS.from_documents(docs, embeddings)

vectorstore = load_vectorstore()

# LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.3
)

question = st.text_input("Ask a question about PragyanAI")

if st.button("Ask"):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    docs = retriever.invoke(question)

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a PragyanAI Student Counselor.

Context:
{context}

Question:
{question}

Answer only using the context provided.
"""

    response = llm.invoke(prompt)

    st.success(response.content)
