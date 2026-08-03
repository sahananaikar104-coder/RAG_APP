
import streamlit as st
import pandas as pd
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="PragyanAI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 PragyanAI Intelligent Assistant")
st.caption("Powered by PragyanAI Knowledge Base + Groq")

# --------------------------------------------------
# PERSONAS
# --------------------------------------------------
PERSONAS = {
    "PragyanAI Student Counselor":
    """
    You are Aarav, an Academic & Career Advisor for PragyanAI.

    Goal:
    Help students understand the PragyanAI program,
    curriculum, fees, placements, projects, hackathons,
    and career opportunities.

    Use ONLY the provided context.
    """,

    "PragyanAI Institutional Advisor":
    """
    You are Dr. Kavita, Institutional Relations Lead.

    Goal:
    Help colleges understand PragyanAI partnerships,
    industry alignment, student outcomes, and AI education.

    Use ONLY the provided context.
    """,

    "PragyanAI Enterprise Lead":
    """
    You are Rohan, Enterprise Placement & Venture Lead.

    Goal:
    Discuss hiring, placements, talent quality,
    AI skills, deployment capabilities, and enterprise value.

    Use ONLY the provided context.
    """
}

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:

    st.header("Settings")

    persona = st.selectbox(
        "Choose Persona",
        list(PERSONAS.keys())
    )

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------
# GROQ KEY
# --------------------------------------------------
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error("GROQ_API_KEY not found in Streamlit Secrets.")
    st.stop()

# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# --------------------------------------------------
# VECTOR STORE
# --------------------------------------------------
@st.cache_resource
def load_vectorstore():

    docs = []

    if not os.path.exists("pragyan_faq_prices.xlsx"):
        st.error(
            "pragyan_faq_prices.xlsx not found in repository."
        )
        st.stop()

    df = pd.read_excel("pragyan_faq_prices.xlsx")

    for _, row in df.iterrows():

        content = " | ".join(
            [f"{col}: {val}" for col, val in row.items()]
        )

        docs.append(
            Document(
                page_content=content,
                metadata={"source": "FAQ"}
            )
        )

    return FAISS.from_documents(
        docs,
        embeddings
    )

vectorstore = load_vectorstore()

# --------------------------------------------------
# LLM
# --------------------------------------------------
@st.cache_resource
def load_llm():

    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )

llm = load_llm()

# --------------------------------------------------
# CHAT MEMORY
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------
question = st.chat_input(
    "Ask anything about PragyanAI..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Thinking..."):

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

        retrieved_docs = retriever.invoke(
            question
        )

        context = "\n\n".join(
            [
                doc.page_content
                for doc in retrieved_docs
            ]
        )

        history_text = "\n".join(
            [
                f"{m['role']}: {m['content']}"
                for m in st.session_state.messages[-8:]
            ]
        )

        prompt = f"""
{PERSONAS[persona]}

Conversation History:
{history_text}

Retrieved Context:
{context}

User Question:
{question}

Rules:
1. Answer only from retrieved context.
2. If information is unavailable, say:
   'I could not find that information in the PragyanAI knowledge base.'
3. Be concise and professional.
"""

        try:

            response = llm.invoke(prompt).content

        except Exception as e:

            response = f"Error: {str(e)}"

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    with st.chat_message("assistant"):
        st.markdown(response)
