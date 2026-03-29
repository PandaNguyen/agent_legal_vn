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

st.set_page_config(page_title="Paralegal AI Assistant", layout="centered", page_icon="⚖️")

# --- Session state ---
if "id" not in st.session_state:
    st.session_state.id = str(uuid.uuid4())[:8]
if "workflow" not in st.session_state:
    st.session_state.workflow = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "workflow_initialized" not in st.session_state:
    st.session_state.workflow_initialized = False


def reset_chat():
    st.session_state.messages = []
    gc.collect()


def render_citations(citations: List[Dict[str, Any]]):
    """Render citation cards cho từng văn bản pháp lý."""
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


@st.cache_resource(show_spinner=False)
def initialize_workflow():
    """Khởi tạo workflow một lần duy nhất, cache lại cho toàn bộ session."""
    embed_data = Embeddata()
    vector_db = QdrantVDB()
    vector_db.initialize_client()
    retriever = Retriever(vector_db=vector_db, embed_data=embed_data, top_k=settings.top_k)
    workflow = AgentWorkflow(
        retriever=retriever,
        gemini_api_key=settings.gemini_api_key,
        llm_model=settings.llm_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )
    return workflow


def run_workflow(inputs: Dict[str, Any]):
    f = io.StringIO()
    with redirect_stdout(f):
        result = st.session_state.workflow.kickoff(inputs)
    return result


# --- Auto-init workflow ---
if st.session_state.workflow is None:
    with st.spinner("Đang khởi động hệ thống..."):
        try:
            st.session_state.workflow = initialize_workflow()
        except Exception as e:
            st.error(f"Không thể khởi tạo hệ thống: {e}")

# --- Main UI ---
col_title, col_btn = st.columns([5, 1])
with col_title:
    st.title("⚖️ Tư vấn Pháp luật AI")
with col_btn:
    st.write("")  # spacing
    if st.button("🗑️ Xóa chat", use_container_width=True):
        reset_chat()
        st.rerun()

st.markdown(
    "Trợ lý AI tra cứu pháp luật Việt Nam — multi-agents · RAG · Web Search",
    unsafe_allow_html=False,
)
st.divider()

# --- Chat history ---
if st.session_state.workflow is not None:
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
                    inputs        = {"query": prompt, "top_k": settings.top_k}
                    response_dict = run_workflow(inputs)

                    if isinstance(response_dict, dict) and "answer" in response_dict:
                        answer    = response_dict["answer"]
                        citations = response_dict.get("citations", [])

                        st.markdown(answer)
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
else:
    st.error("Hệ thống chưa sẵn sàng. Vui lòng kiểm tra cấu hình trong `src/paralegal_agent/config/config.py` và khởi động lại.")