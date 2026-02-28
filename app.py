
import os
import streamlit as st
import gc
import uuid
import time
import io
from contextlib import redirect_stdout

from src.indexing.qdrant_vdb import QdrantVDB
from src.embeddings.embed_data import Embeddata
from src.retrieval.retriever import Retriever
from src.generation.rag import RAG
from src.workflows.agent_workflow import AgentWorkflow
from dotenv import load_dotenv
from config.settings import settings

# Load environment variables
load_dotenv()

# Set up page configuration
st.set_page_config(page_title="Paralegal AI Assistant", layout="wide")

# Initialize session state variables
if "id" not in st.session_state:
    st.session_state.id = str(uuid.uuid4())[:8]

if "workflow" not in st.session_state:
    st.session_state.workflow = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "workflow_logs" not in st.session_state:
    st.session_state.workflow_logs = []

session_id = st.session_state.id


def reset_chat():
    """Reset chat history and clear memory."""
    st.session_state.messages = []
    st.session_state.workflow_logs = []
    gc.collect()


def render_logs(log_text: str):
    """Render logs with ANSI colors and emojis nicely in Streamlit"""
    from ansi2html import Ansi2HTMLConverter
    conv = Ansi2HTMLConverter(inline=True)
    html_body = conv.convert(log_text, full=False)

    st.markdown(
        f"""
        <div style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; white-space: pre-wrap; line-height: 1.45; font-size: 13px;">
        {html_body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def initialize_workflow():
    with st.spinner("🔄 Connecting to backend services..."):
        try:
            # Step 1: Setup embedding model
            st.info("🧠 Loading embedding model...")
            embed_data = Embeddata()
            st.success("✅ Embedding model loaded")

            # Step 2: Setup vector database and connect to Qdrant Cloud
            st.info("🔍 Connecting to Qdrant Cloud...")
            vector_db = QdrantVDB()
            vector_db.initialize_client()
            st.success("✅ Connected to Qdrant Cloud")

            # Step 3: Setup retriever
            retriever = Retriever(
                vector_db=vector_db,
                embed_data=embed_data,
                top_k=settings.top_k
            )
            st.success("✅ Retrieval system ready")

            # Step 4: Setup RAG system
            st.info("🤖 Setting up RAG system...")
            rag_system = RAG(
                retriever=retriever,
                llm_model=settings.llm_model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens
            )
            st.success("✅ RAG system initialized")

            # Step 5: Setup workflow
            st.info("⚙️ Setting up agentic workflow...")
            workflow = AgentWorkflow(
                retriever=retriever,
                rag_system=rag_system,
                firecrawl_apikey=settings.firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY"),
                gemini_api_key=settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
            )

            st.success("🎉 Workflow setup completed!")
            return workflow

        except Exception as e:
            st.error(f"Error initializing workflow: {e}")
            return None


def run_workflow(query: str):
    """Call the synchronous workflow and capture any stdout logs."""
    f = io.StringIO()
    with redirect_stdout(f):
        result = st.session_state.workflow.run_workflow(query)

    logs = f.getvalue()
    if logs:
        st.session_state.workflow_logs.append(logs)

    return result


# Sidebar for configuration
with st.sidebar:
    st.header("🔧 Configuration")

    ollama_model = st.text_input("LLM Model", value="gemma-3-27b-it")
    firecrawl_key = st.text_input("Firecrawl API Key", type="password", value=os.getenv("FIRECRAWL_API_KEY", ""))

    os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "")
    if firecrawl_key:
        os.environ["FIRECRAWL_API_KEY"] = firecrawl_key
        st.success("✅ Firecrawl API Key set!")

    st.markdown("---")

    # Initialize workflow button
    st.header("🚀 System")
    if st.session_state.workflow is None:
        if st.button("▶ Initialize Workflow", use_container_width=True):
            workflow = initialize_workflow()
            if workflow:
                st.session_state.workflow = workflow
                st.balloons()
                st.rerun()
    else:
        st.success("🟢 Workflow is running")
        if st.button("🔄 Re-initialize", use_container_width=True):
            st.session_state.workflow = None
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<p style='font-size:12px; color:#888;'>Data is pre-loaded from static JSON "
        "and embedded in Qdrant Cloud.</p>",
        unsafe_allow_html=True,
    )


# Main chat interface
col1, col2 = st.columns([6, 1])

with col1:
    st.markdown('''
        <h1 style='color: #2E86AB; margin-bottom: 10px;'>
            ⚖️ Paralegal AI assistant
        </h1>
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 20px;">
            <span style='color: #A23B72; font-size: 16px;'>Powered by</span>
            <div style="display: flex; align-items: center; gap: 20px;">
                <a href="#" style="display: inline-block; vertical-align: middle;">
                    <img src="https://images.seeklogo.com/logo-png/61/2/crew-ai-logo-png_seeklogo-619843.png" 
                         alt="CrewAI" style="height: 100px;">
                </a>
                <a href="#" style="display: inline-block; vertical-align: middle;">
                    <img src="https://qdrant.tech/img/logo.svg" 
                         alt="Qdrant" style="height: 32px;">
                </a>
                <a href="#" style="display: inline-block; vertical-align: middle;">
                    <img src="https://i.ibb.co/VcsfddTr/logo-dark.png" 
                         alt="Firecrawl" style="height: 45px;">
                </a>
                <a href="#" style="display: inline-block; vertical-align: middle;">
                    <img src="https://i.ibb.co/wt57zN1/ollama.png" 
                         alt="Ollama" style="height: 48px;">
                </a>
            </div>
        </div>
    ''', unsafe_allow_html=True)

with col2:
    if st.button("Clear Chat ↺", on_click=reset_chat):
        st.rerun()

# System status
if st.session_state.workflow:
    st.success("🟢 System Ready - Workflow initialized successfully!")
else:
    st.info("🔵 Click **Initialize Workflow** in the sidebar to get started")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question..."):
    if not st.session_state.workflow:
        st.error("⚠️ Please initialize the workflow first (see sidebar).")
        st.stop()

    if not os.getenv("GEMINI_API_KEY"):
        st.error("⚠️ Please set your Gemini API key (GEMINI_API_KEY) in the .env file.")
        st.stop()

    # Add user message to chat history
    log_index = len(st.session_state.workflow_logs)
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "log_index": log_index
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        retrieval_time = None

        try:
            with st.spinner("🔄 Processing your query..."):
                workflow_start = time.perf_counter()
                result = run_workflow(prompt)
                workflow_end = time.perf_counter()
                workflow_time = workflow_end - workflow_start

            if isinstance(result, dict) and "answer" in result:
                full_response = result["answer"]

                if result.get("web_search_used", False):
                    st.info("🌐 This response includes information from web search")
                else:
                    st.info("📚 This response is based on the knowledge base")
                    try:
                        retriever = getattr(st.session_state.workflow, "retriever", None)
                        if retriever:
                            retrieve_start = time.perf_counter()
                            retriever.search(prompt)
                            retrieve_end = time.perf_counter()
                            retrieval_time = retrieve_end - retrieve_start

                            citations = retriever.get_citation(prompt, top_k=settings.top_k, snippet_chars=300)

                            if citations:
                                with st.expander("📎 Citations (top matches)"):
                                    for c in citations:
                                        score = c.get("score")
                                        try:
                                            score_str = f"{float(score):.3f}"
                                        except Exception:
                                            score_str = str(score)
                                        st.markdown(
                                            f"[{c['rank']}] score={score_str}"
                                        )
                                        if c.get("snippet"):
                                            st.code(c["snippet"], language="text")
                    except Exception as e:
                        st.warning(f"Could not fetch citations: {e}")

                    if retrieval_time is not None:
                        st.caption(f"🕒 Retrieval time: {retrieval_time:.2f} s")

            else:
                full_response = str(result)

            # Stream the response word by word
            streamed_response = ""
            words = full_response.split()

            for i, word in enumerate(words):
                streamed_response += word + " "
                message_placeholder.markdown(streamed_response + "▌")
                if i < len(words) - 1:
                    time.sleep(0.05)

            message_placeholder.markdown(full_response)

        except Exception as e:
            error_msg = f"❌ Error processing your question: {str(e)}"
            st.error(error_msg)
            full_response = "I apologize, but I encountered an error while processing your question. Please try again."
            message_placeholder.markdown(full_response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666; font-size: 12px;'>"
    "Paralegal AI assistant • Built with Streamlit, CrewAI, Qdrant, Firecrawl, and Ollama"
    "</p>",
    unsafe_allow_html=True
)