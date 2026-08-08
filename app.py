import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1:8b"

SYSTEM_PROMPT = """You are a helpful assistant.
Answer the user's question using ONLY the provided context.
If the answer cannot be found in the context, say:
"I couldn't find that information in the documentation."
Always be concise.
"""

st.set_page_config(page_title="Local RAG", page_icon="📚", layout="wide")


# --- Cached resources so we don't reload models / reopen the DB on every rerun ---
@st.cache_resource
def load_retriever():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    return db.as_retriever(search_kwargs={"k": 5})


@st.cache_resource
def load_llm():
    return ChatOllama(
        model=LLM_MODEL,
        temperature=0,
    )


retriever = load_retriever()
llm = load_llm()


def ask(question: str):
    """Identical logic to query.py's ask() so both tools return the same results."""
    docs = retriever.invoke(question)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    prompt = f"""{SYSTEM_PROMPT}
Context:
{context}
Question:
{question}
Answer:
"""
    response = llm.invoke(prompt)
    return response.content, docs


# --- UI ---
st.title("📚 Local RAG")
st.caption(f"LLM: `{LLM_MODEL}` · Embeddings: `{EMBEDDING_MODEL}` · Chroma dir: `{CHROMA_DIR}`")

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ..., "docs": [...]}

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("docs"):
            with st.expander("Sources"):
                for doc in msg["docs"]:
                    label = doc.metadata.get("relative_path", doc.metadata.get("source"))
                    st.markdown(f"- {label}")

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, docs = ask(question)
        st.markdown(answer)
        if docs:
            with st.expander("Sources"):
                for doc in docs:
                    label = doc.metadata.get("relative_path", doc.metadata.get("source"))
                    st.markdown(f"- {label}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "docs": docs})

with st.sidebar:
    st.header("Options")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    st.markdown(
        "This app mirrors `query.py`: same system prompt, same retriever "
        "(`k=5`), same Chroma DB and Ollama models, so answers should match "
        "the CLI tool exactly (modulo Ollama's own run-to-run variance)."
    )
