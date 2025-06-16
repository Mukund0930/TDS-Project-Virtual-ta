TDS Virtual Teaching Assistant (TA)
This project is a comprehensive Retrieval-Augmented Generation (RAG) system designed to function as a Virtual Teaching Assistant for the "Tools and Data Science" (TDS) course. It leverages course materials, documentation, and historical forum discussions to provide accurate, context-aware answers to student queries.

The system is capable of understanding both text-based and image-based (multimodal) questions, finding relevant information from a curated knowledge base, and generating natural language answers with source citations.

Features
RAG Architecture: Finds relevant information before generating an answer to reduce hallucinations and provide factual responses.
Multisource Knowledge Base: Ingests data from diverse sources, including Discourse forum posts and official Markdown documentation.
Multimodal Queries: Accepts questions containing both text and images (e.g., screenshots of error messages).
Source Citations: Provides direct links to the source documents used to formulate an answer, allowing for verification.
Flexible AI Backend: Architected to work with either OpenAI or Google Gemini models.
REST API: Exposes a clean, easy-to-use API for querying the system.
Project Architecture
The project is divided into two main components: an offline data preprocessing pipeline and an online inference API.

![alt text](https://github.com/m-sehrawat/TDS/assets/110291040/8c7c7295-a226-4556-9a2c-fd1546747d7c)

1. Data Scraper & Preprocessing Pipeline (preprocess.py)
This offline script is responsible for building the knowledge base. It is the foundation upon which the entire RAG system is built.

Key Steps:

Data Ingestion:
Discourse Scraper: Reads exported JSON files from the official course Discourse forum.
Markdown Parser: Reads official course documentation written in Markdown.
Cleaning and Parsing:
HTML content from forum posts is stripped of tags and cleaned to extract pure text.
Metadata such as post author, topic title, and creation date are preserved.
Chunking:
Long documents and forum posts are broken down into smaller, overlapping text "chunks." This is crucial for the retrieval model to find specific, relevant passages.
Embedding Generation:
Each text chunk is converted into a high-dimensional vector representation (an "embedding") using a powerful AI model (e.g., Gemini embedding-001 or OpenAI text-embedding-3-small). These embeddings capture the semantic meaning of the text.
Database Storage:
All chunks, their metadata, and their corresponding embeddings are stored in a local SQLite database (knowledge_base.db), which serves as the project's vector store.
2. RAG Inference API (app.py)
This is the live FastAPI server that receives user queries and generates answers.

Query Lifecycle:

Receive Query: The API accepts a POST request at the /query endpoint containing a user's question and an optional image.
Query Embedding: The incoming question (and any context derived from an image) is converted into an embedding using the same model that built the database. This ensures a meaningful comparison.
Retrieval (The "R" in RAG):
The user's query embedding is compared against all the embeddings stored in the SQLite database.
A cosine similarity search identifies the most semantically similar text chunks from the knowledge base. These are the most relevant pieces of information for answering the question.
Augmentation (The "A" in RAG):
The top-ranked, relevant text chunks are collected and "augmented" by prepending them as context to the user's original question.
Generation (The "G" in RAG):
This combined context-and-question prompt is sent to a powerful Large Language Model (LLM) like Gemini 1.5 Flash or GPT-4o Mini.
The LLM is instructed to generate a final, human-readable answer based only on the provided context, citing the source URLs.
Return Response: The server returns a clean JSON object containing the final answer and a list of source links.
How to Run
Prerequisites
Python 3.8+
Node.js and npm (for evaluation)
An API key for either Google Gemini or OpenAI
1. Setup
# Clone the repository
git clone <your-repo-url>
cd <your-repo-name>

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
Use code with caution.
Bash
2. Configuration
Create a .env file in the root directory and add your API keys and provider choice:

# Set the provider to "gemini" or "openai"
PROVIDER="gemini"

# Add your API keys
GEMINI_API_KEY="your-google-gemini-key..."
OPENAI_API_KEY="your-openai-or-aipipe-key..."
Use code with caution.
Env
3. Build the Knowledge Base
Place your raw data files in the downloaded_threads/ and markdown_files/ directories. Then, run the preprocessing script.

# This must be run first!
python preprocess.py
Use code with caution.
Bash
4. Run the API Server
uvicorn app:app --reload
Use code with caution.
Bash
The API will be available at http://127.0.0.1:8000.

5. Evaluate Performance (Optional)
You can test the quality of the RAG system using promptfoo.

# Make sure the server is running, then in a new terminal:
npx promptfoo eval
Use code with caution.
Bash
Deployed Application
The API is publicly deployed on Render and can be accessed at:
https://tds-project-virtual-ta-jirb.onrender.com
