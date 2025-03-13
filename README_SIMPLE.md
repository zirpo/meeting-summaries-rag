# Simple RAG System for Meeting Summaries

This is a simplified version of the RAG system for meeting summaries. It provides a way to index and query meeting summaries using either OpenAI or local models.

## Setup

1. Create a conda environment:

```bash
conda create -n rag2 python=3.11
conda activate rag2
```

2. Install the required packages:

```bash
pip install langchain langchain-openai langchain-community chromadb pydantic python-dotenv tiktoken
```

3. Set up your OpenAI API key in the `.env` file:

```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Indexing Meeting Summaries

To index your meeting summaries, run:

```bash
python test_rag.py
```

This will process all the meeting summaries in the `meetings` directory, create a vector store, and test a query.

### Querying Meeting Summaries

To query your meeting summaries, run:

```bash
python test_rag_query.py
```

This will run multiple test queries against the vector store.

### Using Local Models with Ollama

To use local models with Ollama, first install Ollama from [https://ollama.ai/](https://ollama.ai/) and pull the required models:

```bash
ollama pull nomic-embed-text
ollama pull llama3
```

Then run:

```bash
python test_rag_ollama.py
```

This will use Ollama for both embeddings and completions.

### Using the Web Interface

To use the web interface, first install Gradio:

```bash
pip install gradio
```

Then run:

```bash
python test_rag_web.py
```

This will launch a web interface where you can enter queries and see the responses. You can also toggle between using OpenAI and local models.

## Adding New Meeting Summaries

You can add new meeting summaries in two ways:

### Manual Creation

Create a Markdown file in the `meetings` directory with the following naming convention:

```
YYYY-MM-DD_MeetingTopic.md
```

For example:
- `2024-03-13_ProjectKickoff.md`
- `2024-03-20_DesignReview.md`

### Using the Create Meeting Script

Use the `create_meeting.py` script to create a new meeting summary file:

```bash
python create_meeting.py "Weekly Team Meeting" --participants "John Doe, Jane Smith, Bob Johnson"
```

This will create a new meeting summary file with today's date and the specified topic and participants.

You can also specify a custom date:

```bash
python create_meeting.py "Client Meeting" --date 2024-04-01 --participants "John Doe, Client Representatives"
```

After adding new meeting summaries, run `python test_rag.py` again to reindex the meeting summaries.

## Customization

You can customize the system by editing the `.env` file:

```
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Vector Database Configuration
CHROMA_PERSIST_DIRECTORY=./data/chroma
DOCUMENT_STORE_DIRECTORY=./meetings

# Chunking Configuration
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Model Configuration
EMBEDDING_MODEL=text-embedding-3-small
COMPLETION_MODEL=gpt-4o-mini

# Local Model Configuration (for future use)
OLLAMA_BASE_URL=http://localhost:11434
LOCAL_EMBEDDING_MODEL=nomic-embed-text
LOCAL_COMPLETION_MODEL=llama3

# Use local models flag (set to true when ready to switch)
USE_LOCAL_MODELS=false
```