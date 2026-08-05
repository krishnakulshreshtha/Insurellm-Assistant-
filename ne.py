
import os
import glob
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.manifold import TSNE
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
import gradio as gr


MODEL = "gpt-4.1-nano"
db_name = "vector_db"
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")



folders = glob.glob("knowledge-base/*")

documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)

print(f"Loaded {len(documents)} documents")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

if Path(db_name).exists():
    vectorstore = Chroma(
        persist_directory=db_name,
        embedding_function=embeddings
    )
else:
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=db_name
    )
llm = ChatOpenAI(temperature=0, model_name=MODEL)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)




SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    return retriever.invoke(question)


def combined_question(
    question: str,
    history: list[dict] | None = None
) -> str:

    if not history:
        return question

    prior = "\n".join(
        m["content"]
        for m in history
        if m.get("role") == "user"
    )

    return f"{prior}\n{question}"


def format_sources(docs: list[Document]) -> str:
    """
    Turn retrieved chunks into a markdown block showing which
    file/doc_type each chunk came from, deduplicated by source file.
    """
    if not docs:
        return "_No sources retrieved for this answer._"

    seen = {}
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        doc_type = doc.metadata.get("doc_type", "unknown")
        seen.setdefault((src, doc_type), []).append(doc.page_content)

    lines = ["**Sources**"]
    for i, ((src, doc_type), chunks) in enumerate(seen.items(), start=1):
        filename = os.path.basename(src)
        preview = chunks[0][:160].replace("\n", " ").strip()
        lines.append(
            f"{i}. **{filename}** _(type: {doc_type})_\n   > {preview}..."
        )
    return "\n\n".join(lines)


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = ""
    sources_md = format_sources(docs)
    for chunk in llm.stream(messages):
        response += chunk.content
        yield response, sources_md
  
def main():
    with gr.Blocks() as demo:
        gr.Markdown("## Insurellm Assistant")
        with gr.Row():
            with gr.Column(scale=2):
                sources_panel = gr.Markdown(
                    value="_Sources for the latest answer will appear here._",
                    label="Sources used for this answer",
                )
            with gr.Column(scale=3):
                gr.ChatInterface(
                    fn=answer_question,
                    type="messages",
                    additional_outputs=[sources_panel],
                )

    demo.launch(inbrowser=True)
 
 
if __name__ == "__main__":
    main()




