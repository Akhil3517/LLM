import asyncio
import hashlib
import io
import time
import requests
from fastapi import FastAPI, HTTPException

# --- Intelligent Document Processing ---
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document

# --- Import from our local modules ---
from schemas import HackRxRequest, HackRxResponse
from config import pc, genai, PINECONE_INDEX_NAME, GEMINI_EMBEDDING_MODEL, GEMINI_GENERATION_MODEL

# --- Third-party Library Imports ---
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- FastAPI App Initialization ---
app = FastAPI(
    title="HackRx 6.0 - High-Accuracy RAG Engine with unstructured.io",
    description="Uses intelligent parsing for superior accuracy and robust error handling."
)

# --- Helper Functions ---

def create_namespace_from_url(url: str) -> str:
    """Creates a unique, deterministic, and valid namespace from a URL."""
    hashed_url = hashlib.sha256(url.encode('utf-8')).hexdigest()
    return f"doc-{hashed_url[:32]}"

async def index_document_if_needed(doc_url: str, namespace: str, index):
    """Checks if a document is indexed and indexes it if not, using unstructured.io."""
    stats = index.describe_index_stats()
    if namespace in stats.namespaces and stats.namespaces[namespace].vector_count > 0:
        print(f"✅ Document {namespace} already indexed. Skipping indexing.")
        return

    print(f"🚨 New document detected. Starting one-time indexing for namespace: {namespace}...")
    start_time = time.time()
    
    # Step 1 & 2: Partitioning and Chunking with unstructured.io
    print("   -> Partitioning with unstructured.io for intelligent chunking...")
    response = requests.get(doc_url)
    response.raise_for_status()
    pdf_file = io.BytesIO(response.content)
    
    # "hi_res" strategy provides the best quality parsing
    elements = partition_pdf(file=pdf_file, strategy="hi_res")
    docs = [Document(page_content=str(el), metadata={"source": doc_url, "type": el.category}) for el in elements]
    print(f"   -> Partitioned into {len(docs)} smart chunks.")

    # Step 3: Embed and Upsert
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i+batch_size]
        batch_content = [doc.page_content for doc in batch_docs]
        response = genai.embed_content(model=GEMINI_EMBEDDING_MODEL, content=batch_content, task_type="retrieval_document")
        embeddings = response['embedding']
        ids = [f"chunk_{i+j}" for j in range(len(batch_docs))]
        metadata = [{"text": doc.page_content} for doc in batch_docs]
        index.upsert(vectors=zip(ids, embeddings, metadata), namespace=namespace)
        print(f"   -> Upserted batch {i//batch_size + 1} into {namespace}")

    end_time = time.time()
    print(f"✅ Indexing complete for {namespace}. Time taken: {end_time - start_time:.2f} seconds.")


async def get_single_answer(question: str, namespace: str, generation_model, safety_settings) -> str:
    """Processes a single question with robust error handling."""
    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        query_embedding = genai.embed_content(model=GEMINI_EMBEDDING_MODEL, content=question, task_type="retrieval_query")['embedding']
        
        query_results = index.query(vector=query_embedding, top_k=5, include_metadata=True, namespace=namespace)
        context = "\n\n---\n\n".join([match['metadata']['text'] for match in query_results['matches']])
        
        prompt = f"""You are a helpful assistant for an insurance policy. Based ONLY on the following CONTEXT, answer the user's QUESTION.

- Answer concisely in 2-3 sentences.
- Focus on the most important conditions, numbers, and limits.
- If the answer is not in the context, state 'The answer could not be found in the provided document.'

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
        
        response = generation_model.generate_content(prompt, safety_settings=safety_settings)

        if response.parts:
            return response.text.strip()
        elif response.prompt_feedback.block_reason:
            reason = response.prompt_feedback.block_reason.name
            print(f"Response blocked for question '{question}'. Reason: {reason}")
            return f"The response was blocked by the safety filter due to: {reason}"
        else:
            return "Received an empty response from the model."

    except Exception as e:
        print(f"An unexpected error occurred for question '{question}': {e}")
        return f"An unexpected error occurred: {e}"

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_submission(request: HackRxRequest):
    doc_url = request.documents[0] if isinstance(request.documents, list) else request.documents
    namespace = create_namespace_from_url(doc_url)
    index = pc.Index(PINECONE_INDEX_NAME)

    await index_document_if_needed(doc_url, namespace, index)

    generation_model = genai.GenerativeModel(GEMINI_GENERATION_MODEL)
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    tasks = [get_single_answer(q, namespace, generation_model, safety_settings) for q in request.questions]
    final_answers = await asyncio.gather(*tasks)
    
    return HackRxResponse(answers=final_answers)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)