import os
from dotenv import load_dotenv

import cohere
from qdrant_client import QdrantClient

from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    function_tool,
    set_tracing_disabled,
    enable_verbose_stdout_logging,
)

# --------------------------------------------------
# ENV SETUP
# --------------------------------------------------
load_dotenv()
set_tracing_disabled(True)
enable_verbose_stdout_logging()

# Required environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not all([GEMINI_API_KEY, COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY]):
    raise ValueError("Missing required environment variables")

# --------------------------------------------------
# MODEL PROVIDER
# --------------------------------------------------
provider = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Correct constructor for OpenAIChatCompletionsModel
model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=provider  # ✅ Use openai_client instead of client
)

# --------------------------------------------------
# COHERE EMBEDDINGS
# --------------------------------------------------
cohere_client = cohere.Client(COHERE_API_KEY)

def get_embedding(text: str) -> list[float]:
    """Return embedding vector from Cohere Embed v3"""
    response = cohere_client.embed(
        model="embed-english-v3.0",
        input_type="search_query",
        texts=[text],
    )
    return response.embeddings[0]

# --------------------------------------------------
# QDRANT
# --------------------------------------------------
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# --------------------------------------------------
# TOOL: RETRIEVE
# --------------------------------------------------
@function_tool
def retrieve(query: str) -> list[str]:
    """Retrieve relevant chunks from Qdrant"""
    embedding = get_embedding(query)

    result = qdrant_client.query_points(
        collection_name="humanoid_ai_book",
        query=embedding,
        limit=5
    )

    return [point.payload["text"] for point in result.points]

# --------------------------------------------------
# AGENT
# --------------------------------------------------
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

# --------------------------------------------------
# RUN AGENT
# --------------------------------------------------
if __name__ == "__main__":
    result = Runner.run_sync(
        agent,
        input="give a all modules of humanoid ai"
    )

    print("\nANSWER:\n")
    print(result.final_output)
