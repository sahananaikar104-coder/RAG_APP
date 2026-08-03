import pandas as pd
import streamlit as st

faq_data = {
    "Category": [
        "Program Overview", "Program Structure", "Program Structure",
        "Pricing & Fees", "Pricing & Fees", "Curriculum & Skills",
        "Curriculum & Skills", "Evaluation & Projects", "Career & Placement",
        "Leadership & Contact"
    ],
    "Question": [
        "What is the total duration and structure of the PragyanAI program?",
        "What happens in Phase 1 (First 6 Months)?",
        "What happens in Phase 2 (12 Months)?",
        "What is the fee structure for the Founding Batch?",
        "What is the salary potential after completing the program?",
        "What modules are covered in Months 1-3 (Foundational Core)?",
        "What modules are covered in Months 4-6 (Advanced Frontier)?",
        "How are students evaluated during the 6-month offline training?",
        "What career tracks or roles are unlocked?",
        "Who leads PragyanAI and how can I contact them?"
    ],
    "Answer": [
        "The PragyanAI AI GenAI program is an 18-month journey comprising 6 Months of Fully Offline Training followed by a 12-Month Internship & Placement Drive.",
        "Phase 1 (6 Months) consists of intensive offline training with half-day classroom sessions, half-day hands-on labs, real-time projects, monthly hackathons, and technical seminars.",
        "Phase 2 (12 Months) includes an extended internship, live client assignments, technical mock interviews, resume building, and startup/product development exposure.",
        "Founding Batch (First 100 students): Initial Training Fee is ₹50,000 + Success Fee of ₹50,000 after placement (Total ₹1,00,000, discounted from standard ₹1,50,000).",
        "Target packages: AI Engineer (₹6–₹15 LPA), GenAI Engineer (₹8–₹18 LPA), and Agentic AI Engineer (₹10–₹25 LPA).",
        "Month 1: Python Full Stack & Analytics. Month 2: Data Science & BI Analytics. Month 3: Machine Learning Frameworks (AutoML, Streamlit deployment).",
        "Month 4: Deep Learning & Computer Vision (CNNs, PyTorch, YOLO). Month 5: NLP & Generative AI (LLMs, RAG, LangChain, Fine-tuning). Month 6: Agentic AI (CrewAI, AutoGen, Multi-Agent Systems, MCP).",
        "Students participate in 1 Technical Seminar per skill (evaluated out of 100 marks) and 1 Skill-wise 48-Hour Hackathon with cash prizes (₹5,000 winner, ₹3,000 runner-up).",
        "7 Multi-Track Pathways: Data Analyst, Data Scientist & ML, AI Engineer, GenAI Engineer, Agentic AI Engineer, Product/MVP Engineer, and Software Engineer.",
        "Led by Sateesh Ambesange (Co-Founder, NITK alumnus, 25+ years IT exp). Phone: +91-9741007422 | Email: sateesh.ambesange@pragyanai.com / pragyan.ai.school@gmail.com"
    ]
}

df = pd.DataFrame(faq_data)
df.to_excel("pragyan_faq_prices.xlsx", index=False)
print("✅ Created 'pragyan_faq_prices.xlsx' with PragyanAI presentation data!")
import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
# ---------------------------------------------------------------------------
# 1. System Prompts specifically grounded in PragyanAI Data
# ---------------------------------------------------------------------------
SALES_PROMPTS = {
    "PragyanAI Student Counselor": """You are Aarav, an Academic & Career Advisor for PragyanAI.
Goal: Guide prospective students to enroll in the 18-Month AI/GenAI Program (6 Month Offline Training + 12 Month Placement Drive).

Strict Rule: Answer pricing, fee structures, curriculum details, and salary potential ONLY based on the Document Context below.

Retrieved Document Context:
{context}

Behavior Guidelines:
1. Be encouraging, empathetic, and focus on practical "builder" skill transformation.
2. Highlight key advantages: 100+ projects, 48-hour hackathons, risk-shared pricing (pay-after-placement success fee), and direct mentorship under Sateesh Ambesange.""",

    "PragyanAI Institutional / CoE Advisor": """You are Dr. Kavita, Institutional Relations Lead at PragyanAI.
Goal: Partner with engineering colleges to solve the education trap and transform students from theory learners into product builders.

Strict Rule: Use the retrieved Context below to cite exact program structures, multi-track career pathways, and evaluation rubrics (seminars, hackathons).

Retrieved Document Context:
{context}

Behavior Guidelines:
1. Maintain an authoritative, industry-oriented tone.
2. Focus on bridging the gap between college curricula and high-value industry roles (Agentic AI, GenAI).""",

    "PragyanAI Enterprise AI & Placement Lead": """You are Rohan, Enterprise Placement & Venture Lead at PragyanAI.
Goal: Connect hiring partners and enterprise leaders with top-tier PragyanAI builders and discuss talent recruitment or custom AI automation.

Strict Rule: Reference exact technical skills (CrewAI, AutoGen, LangChain, RAG, Multi-Agent systems) and portfolio deliverables (GitHub profile, live deployed MVPs) from the context below.

Retrieved Document Context:
{context}

Behavior Guidelines:
1. Confident, direct, and ROI-driven tone.
2. Emphasize that PragyanAI engineers are class-hired builders capable of deploying live applications immediately."""
}
# ---------------------------------------------------------------------------
# 2. Vector Store Indexer (Loads Excel FAQ + PDF Documents)
# ---------------------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = None

def load_documents_into_vectorstore(file_paths=None):
    global vectorstore
    docs = []

    # 1. Process UI file uploads (PDFs or Excel files)
    if file_paths:
        for file in file_paths:
            path = file.name if hasattr(file, 'name') else file
            if path.endswith('.pdf'):
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
            elif path.endswith('.xlsx') or path.endswith('.xls'):
                excel_df = pd.read_excel(path)
                for _, row in excel_df.iterrows():
                    content = " | ".join([f"{col}: {val}" for col, val in row.items()])
                    docs.append(Document(page_content=content, metadata={"source": path}))

    # 2. Automatically load default Excel FAQ if present locally
    if os.path.exists("pragyan_faq_prices.xlsx"):
        excel_df = pd.read_excel("pragyan_faq_prices.xlsx")
        for _, row in excel_df.iterrows():
            content = " | ".join([f"{col}: {val}" for col, val in row.items()])
            docs.append(Document(page_content=content, metadata={"source": "pragyan_faq_prices.xlsx"}))

    # Fallback knowledge base if no files are loaded
    if not docs:
        docs = [
            Document(page_content="PragyanAI Program: 6 Months Offline Training + 12 Months Placement Drive. Led by Sateesh Ambesange."),
            Document(page_content="Founding Batch Fee: ₹50,000 initial training + ₹50,000 success fee post placement.")
        ]

    vectorstore = FAISS.from_documents(docs, embeddings)
    return f"✅ PragyanAI Knowledge Base updated successfully with {len(docs)} document chunks!"

# Build initial index
load_documents_into_vectorstore()
# from google.colab import userdata
# # Retrieve your key
# groq_api_key = userdata.get('GROQ_API_KEY')

groq_api_key = st.secrets["GROQ_API_KEY"]
# ---------------------------------------------------------------------------
# 3. Groq LLM & LCEL RAG Pipeline
# ---------------------------------------------------------------------------

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.3
)

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def create_rag_chain(persona_name: str, retrieved_context: str):
    system_instruction = SALES_PROMPTS.get(
        persona_name,
        SALES_PROMPTS["PragyanAI Student Counselor"]
    ).format(context=retrieved_context)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    return prompt | llm | StrOutputParser()
# ---------------------------------------------------------------------------
# 4. Gradio Callbacks
# ---------------------------------------------------------------------------
def respond(message, history, persona_name):
    if not message.strip():
        return ""

    # Search top relevant snippets
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(message)
    context_str = "\n".join([f"- {doc.page_content}" for doc in relevant_docs])

    session_id = f"pragyan_session_{persona_name.replace(' ', '_')}"
    base_chain = create_rag_chain(persona_name, context_str)

    conversational_chain = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    return conversational_chain.invoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}}
    )

def clear_chat_history(persona_name):
    session_id = f"pragyan_session_{persona_name.replace(' ', '_')}"
    if session_id in store:
        store[session_id].clear()


import streamlit as st

# Set page title and layout
st.set_page_config(page_title="PragyanAI Intelligent Assistant", layout="wide")

# ---------------------------------------------------------------------------
# 1. State Initialization
# ---------------------------------------------------------------------------
# Initialize chat history for personas in session state
if "chat_histories" not in st.session_state:
    st.session_state["chat_histories"] = {}

if "kb_status" not in st.session_state:
    st.session_state["kb_status"] = "PragyanAI presentation FAQ pre-loaded."

# ---------------------------------------------------------------------------
# 2. Header
# ---------------------------------------------------------------------------
st.title("PragyanAI Conversational Sales & FAQ Assistant")
st.markdown("Answers program questions based on the **PragyanAI Presentation & FAQ Sheet**.")

# ---------------------------------------------------------------------------
# 3. Sidebar / Column Layout
# ---------------------------------------------------------------------------
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Configuration")
    
    # Dropdown for Persona selection
    persona_options = list(SALES_PROMPTS.keys())
    selected_persona = st.selectbox(
        "Select PragyanAI Persona",
        options=persona_options,
        index=0 if "PragyanAI Student Counselor" not in persona_options else persona_options.index("PragyanAI Student Counselor")
    )
    
    # Ensure current persona has a dedicated history buffer
    if selected_persona not in st.session_state["chat_histories"]:
        st.session_state["chat_histories"][selected_persona] = []

    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload Additional PDFs or Excel Sheets",
        type=["pdf", "xlsx", "xls"],
        accept_multiple_files=True
    )
    
    # Trigger vectorstore load on file upload
    if uploaded_files:
        status_msg = load_documents_into_vectorstore(uploaded_files)
        st.session_state["kb_status"] = status_msg
        
    st.text_input("Knowledge Base Status", value=st.session_state["kb_status"], disabled=True)

with col2:
    st.subheader("Chat Assistant")

    # Display prior chat messages for the currently selected persona
    current_history = st.session_state["chat_histories"][selected_persona]
    
    for message in current_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input box
    if user_input := st.chat_input("Type your message here..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Append user message to history
        st.session_state["chat_histories"][selected_persona].append({"role": "user", "content": user_input})

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Call your respond function
                response_text = respond(user_input, current_history, selected_persona)
                st.markdown(response_text)
        
        # Append assistant response to history
        st.session_state["chat_histories"][selected_persona].append({"role": "assistant", "content": response_text})

    # Clear memory button
    st.write("")
    if st.button("Clear Memory for Selected Persona", type="secondary"):
        clear_chat_history(selected_persona)
        st.session_state["chat_histories"][selected_persona] = []
        st.rerun()
