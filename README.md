# description

small set of tools to build a local rag with your docs. Great for indexing your .md and .txt files from say your obsidian notes. You can modify the ingest.py for other extensions but you might need to add parsers for them

# install requirements

ensure that you have ollama installed and running. Pull these models

```
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

create a venv and activate it

```
python3 -m venv venv-rag
source venv-rag/bin/activate
```

install langchain + chromadb

```
pip install langchain langchain-community langchain-ollama langchain-chroma chromadb
```
make a directory for the chroma-db

```
mkdir chroma_db
```

clone the repo and edit the ingest.py to point it to your documents directory and run the ingest, this might take a while

```
pyhton3 ingest.py
```

# query the rag

```
python3 query.py
```


