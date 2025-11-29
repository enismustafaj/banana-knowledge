from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from google import genai
from google.genai import types

from database import get_steps

app = FastAPI(title="Gemini Chat Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client()

MODEL_NAME = "gemini-2.5-flash"

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    steps = get_steps("workflow")

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=["""
                You are an assistant that provides help only with workflow operartions.
                Answer the questions of the users only based on the following workflow steps provided to you!
                Use that as your context!
                Here are the steps for the workflow: 
            """,
            ", ".join(map(lambda u: u["text"], steps)),
            "! Provide help with the following question!",
            request.message
        ]
    )
    print(response)

    text_response = response.text

    return ChatResponse(reply=text_response)