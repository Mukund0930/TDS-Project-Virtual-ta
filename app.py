# app.py (Gemini-Only Version)
import os
import json
import sqlite3
import numpy as np
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import aiohttp
import asyncio
import logging
from dotenv import load_dotenv
import uvicorn

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# --- CONSTANTS ---
DB_PATH = "knowledge_base.db"
SIMILARITY_THRESHOLD = 0.50
MAX_RESULTS = 10

# --- GEMINI CONFIGURATION ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY is not set in your .env file. The application cannot start.")
    raise SystemExit("Fatal Error: GEMINI_API_KEY not found.")

# --- GEMINI MODEL MAPPING ---
MODELS = {
    "embedding": "embedding-001",
    "chat": "gemini-1.5-flash",
    "vision": "gemini-1.5-pro"
}

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

class LinkInfo(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]

# --- FastAPI App Initialization ---
app = FastAPI(title="RAG Query API (Gemini Edition)", description="API for querying the RAG knowledge base using Google Gemini")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Functions ---
def get_db_connection():
    try:
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(f"Database file not found at {DB_PATH}. Please run preprocess.py with Gemini first.")
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except (sqlite3.Error, FileNotFoundError) as e:
        logger.error(f"Database connection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- Core Logic ---
def cosine_similarity(vec1, vec2):
    try:
        vec1, vec2 = np.array(vec1, dtype=np.float32), np.array(vec2, dtype=np.float32)
        if np.all(vec1 == 0) or np.all(vec2 == 0): return 0.0
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return 0.0 if norm_product == 0 else dot_product / norm_product
    except Exception:
        logger.error("Error in cosine_similarity", exc_info=True)
        return 0.0

async def get_embedding(text: str, max_retries=3):
    logger.info(f"Getting Gemini embedding for text (length: {len(text)})")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODELS['embedding']}:embedContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {"content": {"parts": [{"text": text}]}}
    
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session, session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["embedding"]["values"]
                else:
                    logger.error(f"Error getting Gemini embedding (status {response.status}, attempt {attempt+1}): {await response.text()}")
                    await asyncio.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"Exception getting Gemini embedding (attempt {attempt+1}): {e}", exc_info=True)
            await asyncio.sleep(2 ** attempt)
    raise HTTPException(status_code=500, detail=f"Failed to get Gemini embedding after {max_retries} attempts.")

async def find_similar_content(query_embedding, conn):
    logger.info("Finding similar content in database")
    results = []
    for table_name in ["discourse_chunks", "markdown_chunks"]:
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name} WHERE embedding IS NOT NULL")
        for chunk in cursor.fetchall():
            try:
                embedding = json.loads(chunk["embedding"])
                similarity = cosine_similarity(query_embedding, embedding)
                if similarity >= SIMILARITY_THRESHOLD:
                    result = dict(chunk)
                    result["similarity"] = float(similarity)
                    result["source"] = "discourse" if "post_id" in result else "markdown"
                    results.append(result)
            except Exception:
                logger.warning(f"Could not process chunk ID {chunk['id']} from {table_name}", exc_info=True)

    results.sort(key=lambda x: x["similarity"], reverse=True)
    logger.info(f"Found {len(results)} relevant results above threshold.")
    return results[:MAX_RESULTS]

async def generate_answer(question, relevant_results):
    logger.info(f"Generating answer for question: '{question[:50]}...' using Gemini")
    context = "\n\n".join([f"Source ({res['source']}):\nURL: {res.get('url', 'N/A')}\nContent: {res['content']}" for res in relevant_results])
    
    prompt = f"""You are an expert assistant. Answer the following question based ONLY on the provided context.
    If the context is insufficient, state that you cannot answer. Your answer must be comprehensive yet concise.
    After your answer, provide a "Sources:" section listing the exact URLs and a brief quote from the text you used.

    Context:
    {context}

    Question: {question}

    Your response format:
    [Your Answer Here]

    Sources:
    1. URL: [exact_url_1], Text: "quote from the source text..."
    2. URL: [exact_url_2], Text: "another quote..."
    """
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODELS['chat']}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2}
    }
    
    try:
        async with aiohttp.ClientSession() as session, session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                raise HTTPException(status_code=response.status, detail=f"Gemini LLM API Error: {await response.text()}")
    except Exception as e:
        logger.error(f"Exception generating answer with Gemini: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate answer from Gemini.")

async def process_multimodal_query(question, image_base64):
    if not image_base64:
        return await get_embedding(question)
    
    logger.info("Processing multimodal query using Gemini Vision")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODELS['vision']}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [
            {"text": f"Describe this image in the context of the question: {question}"},
            {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}
        ]}]
    }

    try:
        async with aiohttp.ClientSession() as session, session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                description = result["candidates"][0]["content"]["parts"][0]["text"]
                combined_query = f"{question}\n\nImage Content: {description}"
                return await get_embedding(combined_query)
            else:
                logger.error(f"Gemini Vision API Error: {await response.text()}. Falling back to text-only.")
                return await get_embedding(question)
    except Exception:
        logger.error("Exception in multimodal query. Falling back to text-only.", exc_info=True)
        return await get_embedding(question)

def parse_llm_response(response_text):
    try:
        answer_part, _, sources_part = response_text.partition("Sources:")
        answer = answer_part.strip()
        links = []
        if sources_part:
            pattern = re.compile(r'URL:\s*(?P<url>http\S+),\s*Text:\s*"(?P<text>[^"]+)"')
            for match in pattern.finditer(sources_part):
                links.append(LinkInfo(url=match.group("url"), text=match.group("text")))
        return QueryResponse(answer=answer, links=links)
    except Exception:
        logger.error("Error parsing LLM response.", exc_info=True)
        return QueryResponse(answer=response_text, links=[])

# --- API Endpoints ---
@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    conn = get_db_connection()
    try:
        query_embedding = await process_multimodal_query(request.question, request.image)
        relevant_results = await find_similar_content(query_embedding, conn)
        
        if not relevant_results:
            return QueryResponse(answer="I couldn't find any relevant information in my knowledge base.", links=[])
        
        llm_response_text = await generate_answer(request.question, relevant_results)
        return parse_llm_response(llm_response_text)
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error("An unexpected error occurred in /query", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")
    finally:
        if conn:
            conn.close()

@app.get("/health")
async def health_check():
    health = {"status": "healthy", "provider": "gemini", "models": MODELS}
    try:
        with get_db_connection() as conn:
            health["database"] = "connected"
            cursor = conn.cursor()
            health["discourse_chunks"] = cursor.execute("SELECT COUNT(*) FROM discourse_chunks").fetchone()[0]
            health["markdown_chunks"] = cursor.execute("SELECT COUNT(*) FROM markdown_chunks").fetchone()[0]
    except Exception as e:
        health["status"] = "unhealthy"
        health["database_error"] = str(e)
    return health

# --- Main Execution ---
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)