# Paralegal Agent

An intelligent, automated Vietnamese legal assistant powered by CrewAI. This agent utilizes a robust Retrieval-Augmented Generation (RAG) pipeline combining hybrid search (dense + sparse), automated response evaluation, and smart web search fallback to deliver accurate, highly reliable, and cited legal answers.

## Quick Start

### 1. Prerequisites

- Python 3.10+
- uv for dependency management
- A running instance of [Qdrant](https://qdrant.tech/) (Local or Cloud)

### 2. Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/PandaNguyen/agent_legal_vn.git
   cd agent_legal_vn
   ```

2. Install dependencies:

   ```bash
   uv sync
   ```

3. Configure your environment variables. Create a `.env` file in the root directory:
   ```bash
   cp .env.example .env
   ```
   _(Ensure you fill in the required API keys in the `.env` file)_

### 3. Running the Agent

You can start the main RAG flow by executing the main script:

```bash
uv run streamlit run app.py
```

## Features

- **Hybrid RAG Pipeline**: Combines dense embeddings (`AITeamVN/Vietnamese_Embedding_v2`) and sparse embeddings (`Qdrant/bm25`) for superior retrieval accuracy using Qdrant.
- **CrewAI Flow Architecture**: Modular agentic workflow handling Retrieval → Generation → Evaluation → Synthesis.
- **Automated Answer Evaluation**: Built-in crew to critique the generated answer before presenting it to the user.
- **Web Search Fallback**: Automatically rewrites bad queries and utilizes [Firecrawl](https://www.firecrawl.dev/) for web-based research if the vector database context is deemed insufficient ("GOOD" vs "web_search" evaluation).
- **Citation & Provenance**: Automatically extracts metadata (Document Number, Title, Issuing Authority, Date) into traceable citations for transparency.
- **Multi-LLM Support**: Easily configurable to run on Gemini, Huggingface endpoints (e.g., Qwen), or local Ollama (e.g., Phi3.5).

## Configuration

The application is highly customizable via the `.env` file (managed by `pydantic-settings`).

| Variable                 | Description                                  | Default                                          |
| ------------------------ | -------------------------------------------- | ------------------------------------------------ |
| `GEMINI_API_KEY`         | Google Gemini API Key                        | _None_                                           |
| `HF_TOKEN`               | Huggingface API Key (for LLM and Embeddings) | _None_                                           |
| `FIRECRAWL_API_KEY`      | API Key for web search fallback              | _None_                                           |
| `QDRANT_API_KEY`         | Qdrant Cloud API Key (if not local)          | _None_                                           |
| `QDRANT_URL`             | URL of the Qdrant instance                   | _None_                                           |
| `LLM_MODEL`              | The LLM model string identifier              | `huggingface/Qwen/Qwen3-4B-Instruct-2507:nscale` |
| `EMBEDDINGS_MODEL`       | Dense embedding model                        | `AITeamVN/Vietnamese_Embedding_v2`               |
| `SPARSE_EMBEDDING_MODEL` | Sparse embedding model                       | `Qdrant/bm25`                                    |

For advanced settings like chunking and retrieval parameters (`top_k`, `chunk_size`, `temperature`), check `src/paralegal_agent/config/config.py`.

## Documentation

- [Architecture Diagram](./agent_flow_plot.html) _(Generate using `agent_flow.plot()` in main.py)_
- [Data Pipeline Overview](./src/paralegal_agent/data_pipeline/README.md) _(Currently manages legal corpus crawling, chunking, and JSONL conversion)_

## Retrieval Evaluation

Performance benchmarks based on **2065 evaluated queries** (F-score calculated with beta = 2.0).

### Hybrid Search Performance

| Top-k | Recall | Precision | F2.0-Score | MRR   |
| ----- | ------ | --------- | ---------- | ----- |
| 1     | 0.587  | 0.677     | 0.596      | 0.677 |
| 3     | 0.758  | 0.300     | 0.566      | 0.752 |
| 5     | 0.806  | 0.195     | 0.480      | 0.762 |
| 10    | 0.865  | 0.106     | 0.344      | 0.768 |
| 20    | 0.903  | 0.057     | 0.220      | 0.770 |

### Dense Search Performance

| Top-k | Recall | Precision | F2.0-Score | MRR   |
| ----- | ------ | --------- | ---------- | ----- |
| 1     | 0.665  | 0.765     | 0.675      | 0.765 |
| 3     | 0.814  | 0.323     | 0.608      | 0.829 |
| 5     | 0.848  | 0.205     | 0.505      | 0.835 |
| 10    | 0.883  | 0.109     | 0.353      | 0.837 |
| 20    | 0.911  | 0.057     | 0.223      | 0.838 |

### Sparse Search Performance

| Top-k | Recall | Precision | F2.0-Score | MRR   |
| ----- | ------ | --------- | ---------- | ----- |
| 1     | 0.381  | 0.438     | 0.386      | 0.438 |
| 3     | 0.552  | 0.218     | 0.412      | 0.521 |
| 5     | 0.618  | 0.148     | 0.367      | 0.536 |
| 10    | 0.694  | 0.085     | 0.276      | 0.546 |
| 20    | 0.757  | 0.047     | 0.182      | 0.550 |

## Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

MIT
