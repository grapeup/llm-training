from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from openai import AzureOpenAI

# FastAPI app initialization
app = FastAPI(title="Simple Chatbot API")

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


@app.post("/chat")
async def chat(message: Message):
    try:
        # Add user message to conversation history
        conversation_history.append({
            "role": "user",
            "content": message.content
        })

        # Prepare messages for Azure OpenAI
        messages = [
                       {"role": "system", "content": "You are a helpful assistant."}
                   ] + conversation_history

        # Get response from Azure OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",  # or your deployed model name
            messages=messages,
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