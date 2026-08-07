from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

CHROMA_DIR = "./chroma_db"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1:8b"

# Load embedding model
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# Open the existing Chroma database
db = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)

retriever = db.as_retriever(
    search_kwargs={"k": 5}
)

# Load the LLM
llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0,
)

SYSTEM_PROMPT = """You are a helpful assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I couldn't find that information in the documentation."

Always be concise.
"""


def ask(question: str):
    docs = retriever.invoke(question)

    context = "\n\n---\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""{SYSTEM_PROMPT}

Context:

{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content, docs


def main():
    print("Local RAG CLI")
    print("Type 'quit' or 'exit' to end.\n")

    while True:
        try:
            question = input("> ").strip()

            if not question:
                continue

            if question.lower() in ("quit", "exit"):
                break

            answer, docs = ask(question)

            print()
            print(answer)
            print()

            print("Sources:")
            for doc in docs:
                print(f" - {doc.metadata.get('relative_path', doc.metadata.get('source'))}")

            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()

