import os
import streamlit as st
import gc
import uuid
import io
from contextlib import redirect_stdout
from typing import Dict, Any, List

from src.paralegal_agent.indexing.qdrant_vdb import QdrantVDB
from src.paralegal_agent.embeddings.embed_data import Embeddata
from src.paralegal_agent.retrieval.retriever import Retriever
from src.paralegal_agent.main import AgentWorkflow
from dotenv import load_dotenv
from src.paralegal_agent.config.config import settings

load_dotenv()

st.set_page_config(page_title="Paralegal AI Assistant", layout="wide", page_icon="⚖️")

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
    st.session_state.messages = []
    st.session_state.workflow_logs = []
    gc.collect()


def render_logs(log_text: str):
    st.markdown(
        f"""<div style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
        'Liberation Mono', 'Courier New', monospace; white-space: pre-wrap;
        line-height: 1.45; font-size: 13px;">{log_text}</div>""",
        unsafe_allow_html=True,
    )


def render_citations(citations: List[Dict[str, Any]]):
    """Render citation cards đẹp cho từng văn bản pháp lý."""
    if not citations:
        return

    st.markdown("#### 📚 Nguồn trích dẫn")
    for c in citations:
        score = c.get("score", 0)
        if score >= 0.8:
            score_color = "#16a34a"
        elif score >= 0.6:
            score_color = "#d97706"
        else:
            score_color = "#6b7280"

        source_url        = c.get("source_url", "")
        unit_title        = c.get("unit_title", "")
        doc_title         = c.get("doc_title", "")
        doc_number        = c.get("doc_number", "")
        doc_type          = c.get("doc_type", "")
        legal_field       = c.get("legal_field", "")
        issuing_authority = c.get("issuing_authority", "")
        issue_date        = c.get("issue_date", "")
        signer            = c.get("signer", "")
        snippet           = c.get("text_snippet", "")

        if source_url:
            title_html = (
                f'<a href="{source_url}" target="_blank" '
                f'style="color:#1d4ed8; text-decoration:none; font-weight:600; font-size:14px;">'
                f'📄 [{doc_number}] {unit_title}</a>'
            )
        else:
            title_html = (
                f'<span style="font-weight:600; font-size:14px;">'
                f'📄 [{doc_number}] {unit_title}</span>'
            )

        tags_html = " ".join([
            f'<span style="background:#e2e8f0; border-radius:8px; padding:2px 8px;">{icon} {val}</span>'
            for icon, val in [
                ("📋", doc_type),
                ("⚖️", legal_field),
                ("🏛️", issuing_authority),
                ("📅", issue_date),
                ("✍️", signer),
            ]
            if val
        ])

        st.markdown(
            f"""
            <div style="border:1px solid #e2e8f0; border-radius:10px; padding:14px 16px;
                        margin-bottom:10px; background:#f8fafc;">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:6px;">
                    <div style="flex:1;">{title_html}</div>
                    <div style="margin-left:12px; flex-shrink:0;">
                        <span style="background:{score_color}; color:white; border-radius:12px;
                                     padding:2px 10px; font-size:12px; font-weight:600;">
                            ⚡ {score:.0%}
                        </span>
                    </div>
                </div>
                <div style="font-size:13px; color:#374151; margin-bottom:6px;">
                    <strong>{doc_title}</strong>
                </div>
                <div style="font-size:12px; color:#6b7280; display:flex; flex-wrap:wrap; gap:6px; margin-bottom:8px;">
                    {tags_html}
                </div>
                <details>
                    <summary style="font-size:12px; color:#6b7280; cursor:pointer;">
                        Xem đoạn trích dẫn...
                    </summary>
                    <div style="margin-top:8px; font-size:13px; color:#4b5563; background:#f1f5f9;
                                border-left:3px solid #94a3b8; padding:8px 12px; border-radius:4px;
                                white-space:pre-wrap;">{snippet}</div>
                </details>
            </div>
            """,
            unsafe_allow_html=True,
        )


def initialize_workflow():
    with st.spinner("Kết nối tới Backend Services..."):
        try:
            settings.llm_model         = st.session_state["llm_model"]     if "llm_model"     in st.session_state else settings.llm_model
            settings.gemini_api_key    = st.session_state["gemini_key"]    if "gemini_key"    in st.session_state else settings.gemini_api_key
            settings.firecrawl_api_key = st.session_state["firecrawl_key"] if "firecrawl_key" in st.session_state else settings.firecrawl_api_key
            settings.temperature       = float(st.session_state["temperature"]) if "temperature" in st.session_state else settings.temperature
            settings.max_tokens        = int(st.session_state["max_tokens"]) if "max_tokens"   in st.session_state else settings.max_tokens
            settings.top_k             = int(st.session_state["top_k"])     if "top_k"         in st.session_state else settings.top_k

            os.environ["GEMINI_API_KEY"]    = settings.gemini_api_key
            os.environ["FIRECRAWL_API_KEY"] = settings.firecrawl_api_key

            st.info(
                f"**Settings đang dùng:**\n"
                f"- Model: `{settings.llm_model}`\n"
                f"- Temperature: `{settings.temperature}`\n"
                f"- Max Tokens: `{settings.max_tokens}`\n"
                f"- Top K: `{settings.top_k}`\n"
                f"- Gemini Key: `{'*' * 8 + settings.gemini_api_key[-4:] if len(settings.gemini_api_key) > 4 else '(empty)'}`\n"
                f"- Firecrawl Key: `{'*' * 8 + settings.firecrawl_api_key[-4:] if len(settings.firecrawl_api_key) > 4 else '(empty)'}`"
            )

            st.info("Loading embedding model...")
            embed_data = Embeddata()
            st.success("Embedding model loaded")

            st.info("Connecting to Qdrant Cloud...")
            vector_db = QdrantVDB()
            vector_db.initialize_client()
            st.success("Connected to Qdrant Cloud")

            retriever = Retriever(vector_db=vector_db, embed_data=embed_data, top_k=settings.top_k)
            st.success("Retrieval system ready")

            st.info("Setting up agentic workflow...")
            workflow = AgentWorkflow(
                retriever=retriever,
                gemini_api_key=settings.gemini_api_key,
                llm_model=settings.llm_model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
            st.success("Workflow setup completed!")
            st.session_state.workflow = workflow

        except Exception as e:
            st.error(f"Error initializing workflow: {e}")
            st.session_state.workflow = None


def run_workflow(inputs: Dict[str, Any]):
    f = io.StringIO()
    with redirect_stdout(f):
        result = st.session_state.workflow.kickoff(inputs)
    logs = f.getvalue()
    if logs:
        st.session_state.workflow_logs.append(logs)
    return result


# --- Sidebar ---
with st.sidebar:
    st.header("Settings Configuration")

    st.subheader("LLM Parameters")
    st.text_input("LLM Model Name", value=settings.llm_model, key="llm_model")
    st.slider("Temperature", min_value=0.0, max_value=1.0, value=float(settings.temperature), step=0.1, key="temperature")
    st.number_input("Max Tokens", min_value=128, max_value=8192, value=int(settings.max_tokens), step=128, key="max_tokens")

    st.subheader("Retriever Settings")
    st.number_input("Top K (Retriever)", min_value=1, max_value=10, value=int(settings.top_k), step=1, key="top_k")

    st.subheader("API Keys (Overrides)")
    st.text_input("Gemini API Key", value=settings.gemini_api_key, type="password", key="gemini_key")
    st.text_input("Firecrawl API Key", value=settings.firecrawl_api_key, type="password", key="firecrawl_key")

    st.markdown("---")

    if st.button("Init / Re-init Workflow", use_container_width=True, type="primary"):
        initialize_workflow()

    if st.button("Clear Chat History", use_container_width=True):
        reset_chat()

    with st.expander("Internal Logs", expanded=False):
        if st.session_state.workflow_logs:
            for log in st.session_state.workflow_logs:
                render_logs(log)
        else:
            st.write("No logs yet.")


# --- Main ---
st.title("Tư vấn Pháp luật AI")
st.markdown("Hệ thống trợ lý AI hỗ trợ tra cứu Pháp luật bằng tiếng Việt sử dụng kiến trúc multi-agents kết hợp RAG và Firecrawl Web Search.")

if st.session_state.workflow is None:
    st.warning("Workflow chưa được khởi tạo. Vui lòng kiểm tra API Keys bên thanh (Sidebar) và nhấn nút **Init / Re-init Workflow** để bắt đầu.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                render_citations(message["citations"])
            col1, col2 = st.columns(2)
            with col1:
                if message.get("rag_response"):
                    with st.expander("RAG Response gốc (Qdrant)"):
                        st.write(message["rag_response"])
            with col2:
                if message.get("web_search_results"):
                    with st.expander("Kết quả Web Search (Firecrawl)"):
                        st.write(message["web_search_results"])

    if prompt := st.chat_input("Nhập câu hỏi pháp lý của bạn vào đây..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Đang tìm kiếm và tổng hợp thông tin..."):
                try:
                    inputs        = {"query": prompt, "top_k": st.session_state.get("top_k", settings.top_k)}
                    response_dict = run_workflow(inputs)

                    if isinstance(response_dict, dict) and "answer" in response_dict:
                        answer    = response_dict["answer"]
                        citations = response_dict.get("citations", [])

                        st.markdown(answer)

                        # Citations hiện ngay dưới câu trả lời
                        if citations:
                            render_citations(citations)

                        st.session_state.messages.append({
                            "role":               "assistant",
                            "content":            answer,
                            "citations":          citations,
                            "rag_response":       response_dict.get("rag_response", ""),
                            "web_search_results": response_dict.get("web_search_results", ""),
                        })
                    else:
                        st.markdown(str(response_dict))
                        st.session_state.messages.append({"role": "assistant", "content": str(response_dict)})

                except Exception as e:
                    st.error(f"Đã xảy ra lỗi hệ thống: {str(e)}")