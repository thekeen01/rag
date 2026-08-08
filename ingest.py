from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DOCUMENTS_DIR = Path("./Documents")
CHROMA_DIR = "./chroma_db"

EMBEDDING_MODEL = "nomic-embed-text"

CHUNK_SIZE = 1250
CHUNK_OVERLAP = 250

SUPPORTED_EXTENSIONS = ("*.md", "*.txt")


def load_documents():
    """Recursively load all supported text documents."""

    documents = []

    for pattern in SUPPORTED_EXTENSIONS:
        for path in DOCUMENTS_DIR.rglob(pattern):
            try:
                text = path.read_text(encoding="utf-8")

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(path),
                            "filename": path.name,
                            "relative_path": str(path.relative_to(DOCUMENTS_DIR)),
                            "extension": path.suffix.lower(),
                        },
                    )
                )

                print(f"Loaded {path.relative_to(DOCUMENTS_DIR)}")

            except Exception as e:
                print(f"Failed to load {path}: {e}")

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return splitter.split_documents(documents)


def main():
    print("Loading documents...")

    documents = load_documents()

    if not documents:
        print("No supported documents found.")
        return

    print(f"\nLoaded {len(documents)} documents")

    print("Splitting documents...")
    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    print("Loading embedding model...")
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
    )

    print("Creating Chroma database...")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    print("\nDone!")
    print(f"Database: {CHROMA_DIR}")
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")


if __name__ == "__main__":
    main()

