import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from openai import AzureOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, StrictFloat


# FastAPI app initialization
app = FastAPI(title="Simple Chatbot API")
collection_name = "training"

# Azure OpenAI client initialization
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-08-01-preview",
    azure_endpoint="https://gu-training-llm.openai.azure.com/"
)

# Store conversation history in memory
conversation_history: List[dict] = []


class Message(BaseModel):
    content: str


qdrant = QdrantClient(":memory:")
qdrant.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
)

def embed(text: str) -> List[float]:
    return client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    ).data[0].embedding

def add_embedding(text: str):
    qdrant.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=[StrictFloat(x) for x in embed(text)],
                payload={
                    'content': text,
                }
            )
        ]
    )

def find_similar_documents(text: str) -> str:
    return str(qdrant.search(
        collection_name=collection_name,
        query_vector=embed(text),
        limit=5,
        query_filter=None
    ))

@app.post("/document")
async def add_document(message: Message):
    add_embedding(message.content)

@app.post("/chat")
async def chat(message: Message):
    try:
        # Add user message to conversation history
        conversation_history.append({
            "role": "user",
            "content": f"""Answer the given question using the context. If there is no answer in the context, say "I don't know". Don't make up the answer instead.
             question: {message.content}
             context: {find_similar_documents(message.content)}"""
        })

        # Prepare messages for Azure OpenAI
        messages = [
                       {"role": "system", "content": "You are a helpful assistant."}
                   ] + conversation_history

        # Get response from Azure OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",  # or your deployed model name
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )

        # Extract assistant's response
        assistant_message = response.choices[0].message.content

        # Add assistant's response to conversation history
        conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        return {
            "response": assistant_message
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)