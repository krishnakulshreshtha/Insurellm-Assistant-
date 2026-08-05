# Insurellm Assistant 🤖

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about **Insurellm** using its internal knowledge base. Built with LangChain, Chroma, HuggingFace embeddings, OpenAI, and a Gradio chat UI.

## How it works

1. **Load** — Markdown files under `knowledge-base/<category>/` are loaded, tagging each document with its `doc_type` (the folder name).
2. **Chunk** — Documents are split into ~1000-character chunks with 200-character overlap using `RecursiveCharacterTextSplitter`.
3. **Embed & Store** — Chunks are embedded with the `all-MiniLM-L6-v2` sentence-transformer model (via `HuggingFaceEmbeddings`) and stored in a local **Chroma** vector database (`vector_db/`).
4. **Retrieve** — On each question, the top 4 most relevant chunks are retrieved from Chroma.
5. **Generate** — The retrieved context is injected into a system prompt and sent to `gpt-4.1-nano` (via `ChatOpenAI`), which streams back an answer.
6. **Chat UI** — A Gradio `ChatInterface` displays the conversation, with a side panel showing the sources (filename + doc type + preview snippet) used for the latest answer.

## Features

- 🔍 Semantic search over a custom markdown knowledge base
- 📚 Source attribution — see exactly which documents informed each answer
- 💬 Streaming responses in a clean chat interface
- 🗂️ Automatic vector store caching — embeddings are only computed once (reused if `vector_db/` already exists)
- 🧠 Follow-up-aware retrieval — prior user turns are folded into the retrieval query for better context

## Project structure

```
.
├── ne.py                  # Main application script
├── knowledge-base/        # Your markdown source documents, organized by category
│   ├── products/
│   ├── employees/
│   └── ...
├── vector_db/              # Persisted Chroma vector store (auto-created)
└── .env                    # Your OpenAI API key (not committed)
```

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install langchain-openai langchain-chroma langchain-huggingface langchain-community \
            langchain-text-splitters scikit-learn python-dotenv gradio
```

### 2. Add your OpenAI API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

### 3. Add your knowledge base

Place your markdown documents inside `knowledge-base/<category>/*.md`, where each subfolder name becomes the `doc_type` metadata tag (e.g. `knowledge-base/products/`, `knowledge-base/employees/`).

### 4. Run the app

```bash
python ne.py
```

This launches a local Gradio app in your browser where you can chat with the assistant.

> **Note:** The first run will build the vector store from your knowledge base and persist it to `vector_db/`. Subsequent runs reuse the existing store — delete the `vector_db/` folder if you update your source documents and want to re-index.

## Tech stack

| Component | Tool |
|---|---|
| LLM | OpenAI `gpt-4.1-nano` |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace) |
| Vector store | Chroma |
| Orchestration | LangChain |
| UI | Gradio |

## Roadmap ideas

- [ ] Add evaluation harness for retrieval quality
- [ ] Support additional file types (PDF, DOCX)
- [ ] Add citation links directly in the chat response
- [ ] Deploy as a hosted demo (e.g. Hugging Face Spaces)


