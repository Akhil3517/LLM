# main.py (Final version with Safety Settings fix)

import requests
import io
import time
import pypdf
from fastapi import FastAPI, HTTPException

# --- Import from our new local modules ---
from schemas import HackRxRequest, HackRxResponse
from config import pc, genai, PINECONE_INDEX_NAME, GEMINI_EMBEDDING_MODEL, GEMINI_GENERATION_MODEL

# --- Third-party Library Imports ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Import the specific types for safety settings
from google.generativeai.types import HarmCategory, HarmBlockThreshold


# --- FastAPI App Initialization ---
app = FastAPI(
    title="HackRx 6.0 with Gemini Pro",
    description="An intelligent query system powered by Google Gemini and Pinecone.",
)

# --- Helper Functions ---
def process_and_chunk_documents(doc_urls: list[str]) -> list[str]:
    """Downloads, parses, and chunks documents."""
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    for url in doc_urls:
        print(f"Downloading and processing: {url}")
        response = requests.get(url)
        response.raise_for_status()
        
        pdf_file = io.BytesIO(response.content)
        pdf_reader = pypdf.PdfReader(pdf_file)
        
        full_text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        chunks = text_splitter.split_text(full_text)
        all_chunks.extend(chunks)
        print(f"Extracted {len(chunks)} chunks from the document.")
    return all_chunks

def get_answers_from_gemini(questions: list[str], chunks: list[str]) -> list[str]:
    """Embeds with Gemini, Retrieves from Pinecone, and Generates Answers with Gemini Pro."""
    
    if not pc or not genai:
        raise HTTPException(status_code=500, detail="API clients not initialized. Check config.py and .env file.")

    index = pc.Index(PINECONE_INDEX_NAME)
    
    print("Embedding and indexing document chunks with Gemini...")
    
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        response = genai.embed_content(model=GEMINI_EMBEDDING_MODEL, content=batch_chunks, task_type="retrieval_document")
        embeddings = response['embedding']
        
        ids = [f"chunk_{j}" for j in range(i, i + len(batch_chunks))]
        metadata = [{"text": chunk} for chunk in batch_chunks]
        
        index.upsert(vectors=zip(ids, embeddings, metadata))
        print(f"Upserted batch {i//batch_size + 1}")
        time.sleep(1)

    print("Indexing complete.")

    final_answers = []
    generation_model = genai.GenerativeModel(GEMINI_GENERATION_MODEL)
    
    # --- FIX: Define safety settings to be more lenient ---
    # This prevents the model from blocking responses due to sensitive (but safe) topics like medical procedures.
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    for q in questions:
        print(f"Processing question: {q}")
        query_embedding = genai.embed_content(model=GEMINI_EMBEDDING_MODEL, content=q, task_type="retrieval_query")['embedding']
        
        query_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        context = "\n\n---\n\n".join([match['metadata']['text'] for match in query_results['matches']])
        
        prompt = f"""Based ONLY on the following context, answer the user's question. If the answer is not present in the context, state 'The answer could not be found in the provided document.' Do not use any external knowledge.

CONTEXT:
{context}

QUESTION:
{q}

ANSWER:"""
        
        # Pass the safety_settings to the generation call
        response = generation_model.generate_content(prompt, safety_settings=safety_settings)
        
        # --- FIX: Add a try-except block for robustness ---
        try:
            answer = response.text.strip()
        except ValueError:
            # This will catch the error if the response is blocked despite our settings.
            answer = "The response was blocked by content safety filters."
            print(f"Response blocked for question: '{q}'. Safety ratings: {response.prompt_feedback}")
            
        final_answers.append(answer)
        print(f"Generated answer: {answer[:80]}...")
        
    return final_answers

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_submission(request: HackRxRequest):
    try:
        if isinstance(request.documents, str):
            doc_urls = [request.documents]
        else:
            doc_urls = request.documents

        all_chunks = process_and_chunk_documents(doc_urls)
        
        if not all_chunks:
            raise HTTPException(status_code=400, detail="Could not extract any text from the documents.")

        final_answers = get_answers_from_gemini(request.questions, all_chunks)
        
        return HackRxResponse(answers=final_answers)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

# --- Main execution block ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)