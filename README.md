# ⚖️ Paralegal AI Assistant

An intelligent legal assistant system that leverages Retrieval-Augmented Generation (RAG) and Agentic Workflows to provide accurate answers and information retrieved from Vietnamese legal documents.

---

## 🌟 Chức năng chính (Features)

- **RAG Pipeline**: Truy xuất thông tin pháp luật chính xác thông qua hệ thống tệp và CSDL Vector.
- **Vector Database**: Tích hợp [Qdrant](https://qdrant.tech/) để lưu trữ và tìm kiếm vector nhúng (embeddings) tốc độ cao.
- **Agentic Workflow**: Vận hành dựa trên [CrewAI](https://www.crewai.com/), kết hợp nhiều công cụ như tìm kiếm web để trả lời câu hỏi phức tạp.
- **Web Search Integration**: Sử dụng [Firecrawl](https://www.firecrawl.dev/) để trích xuất thông tin mới nhất trên web khi CSDL hiện tại không đủ đáp ứng.
- **LLM Providers**: Hỗ trợ Google Gemini thông qua CrewAI.
- **User Interface**: Giao diện trực quan, dễ sử dụng được xây dựng bằng [Streamlit](https://streamlit.io/).
- **Data Web Crawler**: Công cụ thu thập dữ liệu pháp luật với cơ chế tránh bị bot-detection cực tốt dựa trên `curl_cffi` và `asyncio`.

---

## 🛠 Kiến trúc Hệ thống (Architecture)

Project được tổ chức theo module:

- **`app.py`**: Entry point chính của Streamlit UI.
- **`src/`**: Chứa toàn bộ logic lõi:
  - `embeddings/`: Tạo vector nhúng (Embedding generation).
  - `indexing/`: Giao tiếp với Vector DB (Qdrant).
  - `retrieval/`: Logic lấy dữ liệu (Retriever).
  - `generation/`: Logic pipeline RAG chính.
  - `workflows/`: Hệ thống Agentic bằng CrewUI (Ví dụ: `agent_workflow.py`).
- **`data/`**: Chứa dữ liệu tĩnh, các công cụ crawl dữ liệu (`crawl_data.py`, `legal_crawler.py`), và logs.
- **`config/`**: Quản lý cấu hình `pydantic-settings` và parse environment variables.

---

## 🚀 Hướng dẫn Cài đặt & Sử dụng (Installation & Usage)

### Tra cứu yêu cầu hệ thống

- **Python**: `>= 3.13`
- Khuyến nghị sử dụng **[uv](https://github.com/astral-sh/uv)** làm package manager vì project có cấu hình `uv.lock`.

### 1. Cài đặt môi trường

Clone repository về máy và dùng `uv` để cài đặt dependencies:

```bash
git clone https://github.com/PandaNguyen/agent_legal_vn.git

# Cài đặt toàn bộ package thông qua uv
uv sync
```

Hoặc cài trực tiếp bằng `pip`:

```bash
pip install -e .
```

### 2. Cấu hình biến môi trường

Tạo một file `.env` ở thư mục gốc và cung cấp các key sau (xem chi tiết trong `.env.example` nếu có):

```env
GEMINI_API_KEY="your_gemini_api_key"
FIRECRAWL_API_KEY="your_firecrawl_api_key_here"
# QDRANT_URL="your_qdrant_url" (Tuỳ chọn cấu hình kết nối DB bên trong mã hoặc .env)
# QDRANT_API_KEY="your_qdrant_api_key" (Nếu sử dụng Qdrant Cloud)
```

### 3. Khởi chạy Ứng dụng Streamlit

Kích hoạt môi trường (nếu chưa) và chạy:

```bash
uv run streamlit run app.py
```

Ứng dụng sẽ mở ở trình duyệt tại địa chỉ: `http://localhost:8501`.

### 4. Thu thập Dữ liệu (Crawler)

Để crawl thêm văn bản bản án:

```bash
uv run python data/crawl_data.py
```

Dữ liệu crawl sẽ được lưu dưới thư mục `data/` thay vì lưu trong memory, kèm theo file log `crawler.log` ở đó để thao tác gỡ lỗi tiện dụng.

---

## 📚 Công nghệ sử dụng (Tech Stack)

- **Backend & Orchestration**: Python, Streamlit, CrewAI
- **RAG Tools**: QdrantClient, FastEmbed, Sentence-Transformers, Langchain Text Splitters
- **LLM Integration**: Google GenAI
- **Crawling**: Firecrawl, BeautifulSoup4, curl-cffi, lxml

