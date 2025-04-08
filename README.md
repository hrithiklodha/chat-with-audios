# RAG over audio files

RAG app over audio files.
Used-

- AssemblyAI to generate transcripts from audio files.
- LlamaIndex for orchestrating the RAG app.
- Qdrant VectorDB for storing the embeddings.
- Streamlit to build the UI.

## Installation and setup

set tokens in `.env` file

```bash
ASSEMBLYAI_API_KEY
SAMBANOVA_API_KEY
```

**Setup Qdrant VectorDB**

```bash
docker run -p 6333:6333 -p 6334:6334 \
-v $(pwd)/qdrant_storage:/qdrant/storage:z \
qdrant/qdrant
```

**Install Dependencies**:
Ensure you have Python 3.11 or later installed.

```bash
pip install streamlit assemblyai llama-index-vector-stores-qdrant llama-index-llms-sambanovasystems sseclient-py
```

**Run the app**:

Run the app by running the following command:

```bash
streamlit run app.py
```
