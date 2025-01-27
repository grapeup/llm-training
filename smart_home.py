from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from openai import AzureOpenAI
import random
import json

app = FastAPI(title="Simple Chatbot API")

class Message(BaseModel):
    content: str


# Mock functions
def get_temperature(room: str) -> float:
    """Mock function to get temperature in a specific room"""
    return round(random.uniform(18.0, 25.0), 1)

def unlock_door(side: str) -> bool:
    """Mock function to unlock a specific door"""
    return True

def turn_on_ac(desired_temperature: int) -> bool:
    """Mock function to set AC temperature"""
    return True if 16 <= desired_temperature <= 30 else False

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_temperature",
            "description": "Receives a temperature for the given room'",
            "parameters": {
                "type": "object",
                "properties": {
                    "room": {
                        "type": "string",
                        "enum": ["living room", "bedroom", "kitchen", "hall"],
                        "description": "Room name",
                    },
                },
                "required": ["room"],
                "additionalProperties": False,
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "unlock_door",
            "description": "Unlocks a door'",
            "parameters": {
                "type": "object",
                "properties": {
                    "side": {
                        "type": "string",
                        "enum": ["front", "back"],
                        "description": "Side of the house",
                    },
                },
                "required": ["side"],
                "additionalProperties": False,
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "turn_on_ac",
            "description": "Turns on air conditioning for the selected temperature or the previously set temperature'",
            "parameters": {
                "type": "object",
                "properties": {
                    "desired_temperature": {
                        "type": "number",
                        "description": "Target temperature",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        }
    }
]

def handle_tool_call(tool_call):
    arguments = json.loads(tool_call.function.arguments)
    function_name = tool_call.function.name
    print(f"Calling {function_name} with arguments: {arguments}")
    if function_name == "get_temperature":
        result = get_temperature(arguments["room"])
        return f"{result}°C"

    elif function_name == "unlock_door":
        result = unlock_door(arguments["side"])
        return "Door unlocked successfully" if result else "Failed to unlock door"

    elif function_name == "turn_on_ac":
        result = turn_on_ac(arguments["desired_temperature"])
        return "AC temperature set successfully" if result else "Failed to set AC temperature"

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
                       {"role": "system", "content": "You are a helpful home assistant. Your job is to manage smart home using tools."}
                   ] + conversation_history

        call_llm = True

        while call_llm:

            # Get response from Azure OpenAI
            response = client.chat.completions.create(
                model="gpt-4o",  # or your deployed model name
                messages=messages,
                temperature=0.7,
                tools=tools
            )

            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                messages.append({
                    "role": "assistant",
                    "tool_call_id": tool_call.id,
                    "tool_calls": [tool_call]
                })
                function_result = handle_tool_call(tool_call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(function_result)
                })
            else:
                call_llm = False

                # Extract assistant's response
                assistant_message = response.choices[0].message.content

                # Add assistant's response to conversation history
                messages.append({
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