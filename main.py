import os
from dotenv import load_dotenv

import cohere
from qdrant_client import QdrantClient

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    function_tool,
    set_tracing_disabled,
    enable_verbose_stdout_logging,
)

# ---------------- ENV SETUP ----------------
load_dotenv()
set_tracing_disabled(True)
enable_verbose_stdout_logging()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

missing_vars = []
for key, value in {
    "GEMINI_API_KEY": GEMINI_API_KEY,
    "COHERE_API_KEY": COHERE_API_KEY,
    "QDRANT_URL": QDRANT_URL,
    "QDRANT_API_KEY": QDRANT_API_KEY,
}.items():
    if not value:
        missing_vars.append(key)

if missing_vars:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")

# ---------------- MODEL PROVIDER ----------------
provider = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=provider
)

# ---------------- COHERE EMBEDDINGS ----------------
cohere_client = cohere.Client(COHERE_API_KEY)

def get_embedding(text: str) -> list[float]:
    try:
        response = cohere_client.embed(
            model="embed-english-v3.0",
            input_type="search_query",
            texts=[text],
        )
        return response.embeddings[0]
    except Exception as e:
        print("Cohere embedding error:", e)
        return []

# ---------------- QDRANT ----------------
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# ---------------- TOOL: RETRIEVE ----------------
@function_tool
def retrieve(query: str) -> list[str]:
    try:
        embedding = get_embedding(query)
        if not embedding:
            return []
        result = qdrant_client.query_points(
            collection_name="humanoid_ai_book",
            query=embedding,
            limit=5
        )
        return [point.payload.get("text", "") for point in result.points]
    except Exception as e:
        print("Qdrant retrieve error:", e)
        return []

# ---------------- AGENT ----------------
agent = Agent(
    name="Physical AI Tutor",
    instructions="""
You are an AI tutor for the Physical AI & Humanoid Robotics textbook.

Rules:
1. First call the tool `retrieve` with the user question.
2. Use ONLY the returned content from `retrieve` to answer.
3. If the answer is not present, reply exactly: "I don't know".
""",
    model=model,
    tools=[retrieve]
)

# ---------------- FASTAPI SETUP ----------------
app = FastAPI(title="Humanoid AI RAG Agent")

# ---------------- CORS SETUP ----------------
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:3000",
    "https://hackathon-eight-beige.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- API ----------------
class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask_agent(req: QueryRequest):
    try:
        if not req.query.strip():
            return {"answer": "Query is empty."}
        result = Runner.run_sync(agent, input=req.query)
        return {"answer": result.final_output or "I don't know"}
    except Exception as e:
        print("Agent error:", e)
        raise HTTPException(status_code=500, detail="Internal server error. Check backend logs.")

@app.get("/")
def root():
    return {"message": "Humanoid AI RAG Agent is running"}
