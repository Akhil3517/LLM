import asyncio
import hashlib
import io
import requests
from fastapi import FastAPI, HTTPException

import PyPDF2
from langchain_core.documents import Document

# Local Modules
from schemas import HackRxRequest, HackRxResponse
from config import pc, genai, PINECONE_INDEX_NAME, GEMINI_EMBEDDING_MODEL, GEMINI_GENERATION_MODEL

# Google API
from google.generativeai.types import HarmCategory, HarmBlockThreshold

app = FastAPI(title="HackRx 6.0 - Fast & Corrected RAG Engine")

def create_namespace_from_url(url: str) -> str:
    hashed_url = hashlib.sha256(url.encode('utf-8')).hexdigest()
    return f"doc-{hashed_url[:32]}"

async def index_document_if_needed(doc_url: str, namespace: str, index):
    stats = index.describe_index_stats()
    if namespace in stats.namespaces and stats.namespaces[namespace].vector_count > 0:
        print(f"✅ Document {namespace} already indexed.")
        return

    print(f"🚨 New document detected. Indexing namespace: {namespace}...")
    response = requests.get(doc_url)
    response.raise_for_status()
    
    # Use PyPDF2 for lighter PDF processing
    pdf_file = io.BytesIO(response.content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    
    docs = []
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        if text.strip():  # Only add non-empty pages
            docs.append(Document(page_content=text, metadata={"source": doc_url, "page": page_num + 1}))

    batch_size = 50  # Reduced batch size for memory optimization
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i+batch_size]
        try:
            response = genai.embed_content(model=GEMINI_EMBEDDING_MODEL, content=[d.page_content for d in batch_docs], task_type="retrieval_document")
            index.upsert(vectors=zip([f"chunk_{i+j}" for j in range(len(batch_docs))], response['embedding'], [{"text": d.page_content} for d in batch_docs]), namespace=namespace)
            print(f"   -> Upserted batch {i//batch_size + 1} into {namespace}")
        except Exception as e:
            print(f"   -> Error in batch {i//batch_size + 1}: {e}")
            continue

async def expand_query(question: str, generation_model):
    prompt = f"Rewrite the following user question to be more detailed for searching an insurance policy. Question: {question}"
    try:
        response = generation_model.generate_content(prompt)
        # We add this check to ensure we don't return an empty or blocked response
        if response.parts:
            return response.text.strip()
    except Exception:
        pass # If expansion fails, we'll just use the original question.
    return question

async def get_single_answer(question: str, namespace: str, generation_model, safety_settings) -> str:
    try:
        index = pc.Index(PINECONE_INDEX_NAME)

        # Step 1: Expand Query (this happens inside the concurrent task)
        expanded_question = await expand_query(question, generation_model)
        
        # Step 2: Embed
        query_embedding = genai.embed_content(model=GEMINI_EMBEDDING_MODEL, content=expanded_question, task_type="retrieval_query")['embedding']
        
        # Step 3: Query
        query_results = index.query(vector=query_embedding, top_k=5, include_metadata=True, namespace=namespace)
        context = "\n\n---\n\n".join([match['metadata']['text'] for match in query_results['matches']])
        
        # Step 4: Generate Final Answer
        prompt = f"""Based ONLY on the CONTEXT, answer the QUESTION concisely. Focus on numbers and key conditions. If the answer is not in the context, state 'The answer could not be found in the provided document.'

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
        
        response = generation_model.generate_content(prompt, safety_settings=safety_settings)

        if response.parts:
            return response.text.strip()
        elif response.prompt_feedback.block_reason:
            return f"Response blocked due to: {response.prompt_feedback.block_reason.name}"
        else:
            return "Received an empty response from the model."

    except Exception as e:
        # This will catch errors like the 400 Bad Request and others.
        print(f"An unexpected error occurred for question '{question}': {e}")
        return f"An unexpected error occurred: {e}"

@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_submission(request: HackRxRequest):
    # This endpoint now correctly orchestrates the concurrent tasks.
    doc_url = request.documents[0] if isinstance(request.documents, list) else request.documents
    namespace = create_namespace_from_url(doc_url)
    index = pc.Index(PINECONE_INDEX_NAME)

    await index_document_if_needed(doc_url, namespace, index)

    generation_model = genai.GenerativeModel(GEMINI_GENERATION_MODEL)
    
    # ## FIX FOR THE 400 BAD REQUEST ##
    # Only specify the four user-configurable safety categories.
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    # ## FIX FOR THE 4-MINUTE RUNTIME ##
    # Create a list of tasks, where each task is a complete get_single_answer call.
    # This ensures that everything (expansion, query, generation) runs in parallel for each question.
    tasks = [get_single_answer(q, namespace, generation_model, safety_settings) for q in request.questions]
    
    # Run all question-answering tasks concurrently.
    final_answers = await asyncio.gather(*tasks)
    
    return HackRxResponse(answers=final_answers)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)