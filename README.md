# Smart Librarian – AI Book Recommender

This project implements a smart AI-powered book recommendation chatbot using OpenAI GPT-4o-mini, function calling and semantic search via a local vector store (ChromaDB). The chatbot suggests books based on user input and provides a summary using a custom tool function.

## Features Implemented 

- Book summary database with 10+ structured summaries stored locally
- Semantic search using ChromaDB and OpenAI embeddings (text-embedding-3-small)
- Retriever-based RAG setup that finds best book matches based on user input
- GPT-4o-mini integration via function calling (chat + tool round)
- Registered tool function `get_summary_by_title(title: str)` returning full local summary
- Function calling flow that forces the model to call the tool, then format a final answer
- Filter for off-topic or irrelevant questions using keyword match + semantic distance
- `.env` removed — OpenAI API key is injected via environment variable (OPENAI_API_KEY)
- FastAPI backend with endpoint `/ask` to test book recommendations


## Example CLI Usage 

### Valid question (returns a book recommendation):

```
curl -X POST http://127.0.0.1:8000/ask ^
-H "Content-Type: application/json" ^
-d "{ \"question\": \"I want a book about friendship and magic\" }"
```
<img width="902" height="668" alt="image" src="https://github.com/user-attachments/assets/32854250-cbfb-4525-8b37-249130a7f08a" />

### Off-topic question (rejected politely):

```
curl -X POST http://127.0.0.1:8000/ask ^
-H "Content-Type: application/json" ^
-d "{ \"question\": \"What is the price of an apple?\" }"
```
<img width="888" height="565" alt="image" src="https://github.com/user-attachments/assets/3f2c7864-dca1-412d-82ad-b912fb8e7032" />


## Installation

1. Clone the project:

```
git clone https://github.com/ioanamarc/SLibrarian.git
cd SLibrarian
```

2. Create virtual environment and install dependencies:

```
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

3. Set your OpenAI API key via environment variable:

Example for PowerShell:
```
$env:OPENAI_API_KEY="sk-..."
```

## Running the API

```
uvicorn main:app --reload
```

Access documentation at: http://127.0.0.1:8000/docs
## Notes

- You must set the `OPENAI_API_KEY` environment variable before running the app.
- Tool calling is enforced — the assistant always uses `get_summary_by_title` before answering.
- Retrieval is performed via ChromaDB using OpenAI embedding vectors.
- Off-topic questions are blocked using a combined semantic and keyword-based filter.
