import os
import time
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

def setup_pinecone():
    """Checks if the Pinecone index exists and creates it if it doesn't using ServerlessSpec."""
    load_dotenv()

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = "hackrx-gemini-index"
    PINECONE_VECTOR_DIMENSION = 768 
    CLOUD_PROVIDER = "aws"
    CLOUD_REGION = "us-east-1"

    if not PINECONE_API_KEY:
        print("Error: Pinecone API key not found in .env file.")
        return

    print("Initializing Pinecone client...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    print(f"Found existing indexes: {existing_indexes}")

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Index '{PINECONE_INDEX_NAME}' not found. Creating a new serverless index...")
        
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_VECTOR_DIMENSION,
            metric='cosine',
            spec=ServerlessSpec(
                cloud=CLOUD_PROVIDER,
                region=CLOUD_REGION
            )
        )
        
        print(f"Index creation request sent. It may take a minute to initialize.")
        while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
            print("Waiting for index to be ready...")
            time.sleep(10)
        
        print(f"Index '{PINECONE_INDEX_NAME}' is now ready.")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists. No action needed.")

if __name__ == "__main__":
    setup_pinecone()