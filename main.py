from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import httpx
from typing import Optional
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from textwrap import shorten

# Load environment variables

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

EMBEDDING_URL = "https://aipipe.org/openai/v1/embeddings"
CHAT_URL = "https://aipipe.org/openai/v1/chat/completions"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
QDRANT_COLLECTION = "tds_kb"

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.get("/debug/env")
def check_env():
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
        "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY")
    }

# Initialize Qdrant client
client = QdrantClient(
    url="https://5746a4d0-16c1-4e22-b0c1-3be2674559c5.us-west-2-0.aws.cloud.qdrant.io:6333",
    api_key=QDRANT_API_KEY,
)

# Request payload
class QueryPayload(BaseModel):
    question: str
    image: Optional[str] = None  # Base64 user image

# Function to get embedding
async def get_embedding(text: str):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"model": EMBEDDING_MODEL, "input": text}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(EMBEDDING_URL, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

# Function to generate GPT answer with multiple images
async def generate_gpt_answer(question: str, context: str, images: list):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    system_prompt = """
    You are a helpful assistant that answers academic and technical questions using the context provided.
    Guidelines:
    - `"answer"` should contain a concise and accurate response to the user's question.
    - If an image is provided, use it along with the text context to answer.
    - If a tool or model is mentioned, explain its usage and whether it's supported.
    - If one or more tools or models are mentioned, explain whether it can be used or not.
    - If the user asks for a specific model , always clarify whether that model is available or not.
    -If the user asks about model, clarify its intended use and explain whether it is supported in this setup 
    """

    message_content = [
        {"type": "text", "text": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    for img in images:
        message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img}"}
        })

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message_content}
    ]

    payload = {"model": CHAT_MODEL, "messages": messages}

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(CHAT_URL, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

# Simple summarizer function
def summarize_text(text, max_len=120):
    return shorten(text.strip().replace("\n", " "), width=max_len, placeholder="...")

@app.post("/api")
async def query_api(payload: QueryPayload):
    question = payload.question.strip()
    if not question:
        return {"answer": "Invalid input: Question cannot be empty.", "links": []}

    try:
        query_vector = await get_embedding(question)
    except Exception as e:
        return {"answer": f"Embedding failed: {str(e)}", "links": []}

    try:
        results = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=7
        )
    except Exception as e:
        return {"answer": f"Qdrant search failed: {str(e)}", "links": []}

    if not results:
        return {"answer": "No relevant results found.", "links": []}

    context_blocks = []
    context_images = []
    links = []

    for r in results:
        text = r.payload.get("text", "").strip()
        context_blocks.append(text)
        short_text = summarize_text(text)

        if r.payload.get("source") == "thread":
            images_base64 = r.payload.get("images_base64", [])
            if isinstance(images_base64, list):
                context_images.extend(images_base64)

            if "post_url" in r.payload:
                links.append({
                    "url": r.payload["post_url"],
                    "text": short_text
                })
        else:
            links.append({
                "url": r.payload.get("url", ""),
                "text": short_text
            })

    if payload.image:
        context_images.insert(0, payload.image)

    full_context = "\n\n".join(context_blocks)

    try:
        gpt_answer = await generate_gpt_answer(payload.question, full_context, context_images)
    except Exception as e:
        return {"answer": f"GPT generation failed: {str(e)}", "links": links}

    return {
        "answer": gpt_answer.strip(),
        "links": links
    }
