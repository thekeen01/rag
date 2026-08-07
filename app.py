import streamlit as st

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

CHROMA_DIR = "./chroma_db"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1:8b"

st.set_page_config(
    page_title="Local RAG",
    page_icon="📚",
    layout="wide",
)

# Load models once
@st.cache_resource
def load_rag():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    retriever = db.as_retriever(search_kwargs={"k": 5})

    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0,
    )

    return retriever, llm


retriever, llm = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("📚 Local RAG")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask your documentation...")

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    docs = retriever.invoke(prompt)

    context = "\n\n---\n\n".join(
        d.page_content for d in docs
    )

    rag_prompt = f"""
You are answering questions using the supplied documentation.

If the answer is not present, say so.

Documentation:

{context}

Question:

{prompt}
"""

    response = llm.invoke(rag_prompt)

    answer = response.content

    with st.chat_message("assistant"):
        st.markdown(answer)

        with st.expander("Sources"):
            for doc in docs:
                st.write(doc.metadata.get("relative_path"))
                st.caption(doc.page_content[:300] + "...")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

