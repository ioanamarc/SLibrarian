import os
import traceback
from typing import List, Dict, Any

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from openai import OpenAI

# Local data
from book_summaries import book_summaries_dict, get_summary_by_title

# =========================
# Environment & clients
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Please set it as an environment variable.")

# OpenAI client
oai = OpenAI(api_key=OPENAI_API_KEY)

# Chroma client
chroma_client = chromadb.Client()

# Embedding function
embedding_fn = OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small",
)


# Create the collection that will hold our book summaries
collection = chroma_client.get_or_create_collection(
    name="book_summaries",
    embedding_function=embedding_fn,
)


# =========================
# Data loading into Chroma
# =========================
def _ensure_collection_populated() -> None:
    """
    Idempotently (re)populate the Chroma collection with our known set of titles + summaries.
    Safe to call multiple times; existing ids will be upserted.
    """
    ids, docs, metadatas = [], [], []
    for idx, (title, summary) in enumerate(book_summaries_dict.items()):
        ids.append(f"book-{idx}")
        docs.append(summary)
        metadatas.append({"title": title})

    collection.upsert(ids=ids, documents=docs, metadatas=metadatas)


_ensure_collection_populated()


# =========================
# FastAPI models
# =========================
class AskRequest(BaseModel):
    question: str
    top_k: int = 3


class AskResponse(BaseModel):
    recommended_title: str
    full_summary: str
    message: str
    candidates: List[Dict[str, Any]]


# =========================
# FastAPI app
# =========================
app = FastAPI(title="Books RAG Chat", version="1.0.0")


from starlette.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


# =========================
# OpenAI function-tool schema
# =========================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_summary_by_title",
            "description": "Return the full summary for an exact book title from the local database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Exact book title to fetch. Must be one of the provided titles in context."
                    }
                },
                "required": ["title"],
                "additionalProperties": False
            },
        },
    }
]


SYSTEM_PROMPT = (
    "You are a helpful assistant that recommends a single book given a user question. "
    "You are connected to a retrieval system that provides a short list of candidate titles. "
    "Steps:\n"
    "1) Read the user's question and the retrieved candidates (titles + excerpts).\n"
    "2) Choose exactly ONE title that best matches the user's needs.\n"
    "3) ALWAYS call the tool get_summary_by_title with the exact title you chose.\n"
    "4) After the tool returns the full summary, craft a concise conversational answer in English:\n"
    "   - Start with the recommendation: the title and 1–2 sentence rationale tailored to the user.\n"
    "   - Then include the full summary returned by the tool under a heading 'Full summary'.\n"
    "   - Keep the tone friendly and clear.\n"
    "If the question is off-topic or unsafe, be brief and decline.\n"
)


def _retrieve_candidates(question: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Query Chroma for the nearest summaries, returning a list of candidate dicts:
    [{'title': ..., 'excerpt': ..., 'distance': ...}, ...]
    """
    top_k = max(1, min(top_k, 5))
    result = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=['metadatas', 'documents', 'distances'],
    )
    candidates: List[Dict[str, Any]] = []
    if result and result.get("metadatas"):
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            excerpt = (doc or "")[:220]
            candidates.append({"title": meta.get("title", ""), "excerpt": excerpt, "distance": float(dist)})
    return candidates


BOOK_KEYWORDS_EN = [
    "book", "books", "novel", "story", "author", "reading", "recommend",
    "library", "fantasy", "adventure", "war", "classic", "thriller", "mystery",
]


def _keyword_hit(q: str) -> bool:
    q = q.lower()
    return any(k in q for k in BOOK_KEYWORDS_EN)


def is_question_about_books(question: str, threshold: float = 1.5) -> bool:
    """
    1) If the question contains book-related keywords -> immediate True.
    2) Otherwise, semantic fallback: if the embedding distance < threshold, treat as 'about books'.
    A slightly relaxed threshold (1.5) helps avoid false negatives.
    """
    try:
        if _keyword_hit(question):
            return True

        result = collection.query(
            query_texts=[question],
            n_results=1,
            include=["distances"],
        )
        dist = (
            float(result["distances"][0][0])
            if result and result.get("distances") and result["distances"][0]
            else float("inf")
        )
        return dist < threshold
    except Exception as e:
        print(f"is_question_about_books error: {e}")
        return True  # do not block the flow on errors


def _tool_call_round(question: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run a function-calling round where the model MUST call get_summary_by_title.
    Then execute the tool locally and run a second round to produce the final answer.
    Returns a dict: {"recommended_title": ..., "full_summary": ..., "message": ...}
    """

    ctx_lines = []
    for i, c in enumerate(candidates, 1):
        ctx_lines.append(f"{i}. {c['title']}: {c['excerpt']}")
    context_block = "Candidates:\n" + "\n".join(ctx_lines) if ctx_lines else "Candidates: (none)"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "system", "content": context_block},
    ]

    first = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
        tool_choice={"type": "function", "function": {"name": "get_summary_by_title"}},
        temperature=0.4,
    )

    choice = first.choices[0]
    tool_calls = choice.message.tool_calls or []
    if not tool_calls:
        # Model didn't follow instructions; fall back to the top candidate
        fallback_title = candidates[0]["title"] if candidates else ""
        summary = get_summary_by_title(fallback_title) if fallback_title else ""
        # Second round to format a nice answer
        tool_messages = messages + [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "manual_fallback", "type": "function", "function": {"name": "get_summary_by_title", "arguments": f'{{"title": "{fallback_title}"}}'}}
                ],
                "content": None,
            },
            {"role": "tool", "tool_call_id": "manual_fallback", "name": "get_summary_by_title", "content": summary},
        ]
        second = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=tool_messages,
            temperature=0.4,
        )
        final_text = second.choices[0].message.content or ""
        return {"recommended_title": fallback_title, "full_summary": summary, "message": final_text}

    # Execute exactly one tool call (the first)
    tc = tool_calls[0]
    args = tc.function.arguments
    # Tool call args come as a JSON string in most SDKs
    import json
    try:
        parsed = json.loads(args) if isinstance(args, str) else args
    except Exception:
        parsed = {}

    title = (parsed or {}).get("title", "")
    summary = get_summary_by_title(title) if title else ""

    # Second round: provide the tool result so the model can write the final answer
    messages_with_tool = messages + [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": "get_summary_by_title", "arguments": args}},
            ],
            "content": None,
        },
        {"role": "tool", "tool_call_id": tc.id, "name": "get_summary_by_title", "content": summary},
    ]

    second = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_with_tool,
        temperature=0.4,
    )
    final_text = second.choices[0].message.content or ""
    return {"recommended_title": title, "full_summary": summary, "message": final_text}


# =========================
# /ask endpoint
# =========================
@app.post("/ask", response_model=AskResponse, responses={500: {"model": None}})
def ask(req: AskRequest):
    try:
        # Retrieve candidates
        candidates = _retrieve_candidates(req.question, req.top_k)

        # Check book relevance
        if not is_question_about_books(req.question):
            return {
                "recommended_title": "",
                "full_summary": "",
                "message": "Your question doesn't seem to be about books. Please ask something related to reading recommendations.",
                "candidates": candidates,
            }

        # Check candidate quality (avoid recommending when semantic distance is too high)
        if not candidates or candidates[0]["distance"] > 1.4:
            return {
                "recommended_title": "",
                "full_summary": "",
                "message": "Sorry, I couldn't find any relevant book for your request.",
                "candidates": candidates,
            }

        # Normal flow
        result = _tool_call_round(req.question, candidates)
        return {
            "recommended_title": result["recommended_title"],
            "full_summary": result["full_summary"],
            "message": result["message"],
            "candidates": candidates,
        }

    except Exception:
        tb = traceback.format_exc()
        return PlainTextResponse(content=tb, status_code=500)
