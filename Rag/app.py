import os
import tempfile
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# استدعاء الموديولات النظيفة التي أنشأناها
from rag.pdf_loader import DocumentProcessor
from rag.retriever import HybridReRankRetriever
from rag.agent_graph import RAGAgentGraph

st.set_page_config(
    page_title="Multimodal Hybrid RAG Agent",
    page_icon="🧠",
    layout="wide"
)

st.title("Enterprise Hybrid RAG Assistant")
st.caption("Modular Pipeline powered by Yassin Sanad")

# --- إدارة الـ Session States ---
if "rag_agent" not in st.session_state:
    st.session_state.rag_agent = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- الـ Sidebar للتحكم ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if not groq_api_key:
        groq_api_key = st.text_input("Enter Groq API Key:", type="password")

    st.markdown("---")
    st.header(" Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

    if uploaded_file and groq_api_key:
        if st.button("Index Document", type="primary"):
            with st.spinner("Processing PDF, indexing vectors, and compiling LangGraph..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                # 1. Processing
                processor = DocumentProcessor()
                docs = processor.process_pdf(tmp_path)
                
                # 2. Retrieving & Graph Setup
                retriever = HybridReRankRetriever(docs)
                st.session_state.rag_agent = RAGAgentGraph(retriever, groq_api_key)
                st.session_state.messages = []
                
                os.remove(tmp_path)
                st.success(f"Indexed successfully! Total chunks: {len(docs)}")

# --- واجهة المحادثة ---
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

if user_prompt := st.chat_input("Ask something about your document..."):
    if not st.session_state.rag_agent:
        st.warning("Please upload a PDF and click 'Index Document' first.")
    else:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append(HumanMessage(content=user_prompt))

        with st.chat_message("assistant"):
            with st.spinner("Processing through LangGraph Pipeline..."):
                config = {"configurable": {"thread_id": "streamlit_session"}}
                inputs = {"messages": [HumanMessage(content=user_prompt)]}
                
                result = st.session_state.rag_agent.graph.invoke(inputs, config=config)
                response_text = result["messages"][-1].content
                
                st.markdown(response_text)
                st.session_state.messages.append(AIMessage(content=response_text))
