from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Travel Planner AI")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up templates
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Sample data for the landing page
POPULAR_DESTINATIONS = [
    {"name": "Araku Valley", "image": "araku.jpg", "description": "Scenic hill station with coffee plantations"},
    {"name": "Kodaikanal", "image": "kodaikanal.jpg", "description": "Princess of Hill Stations"},
    {"name": "Munnar", "image": "munnar.jpg", "description": "Tea gardens and misty mountains"},
]

class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "AI Travel Planner - Plan Your Perfect Trip",
            "popular_destinations": POPULAR_DESTINATIONS
        }
    )

@app.get("/chat", response_class=HTMLResponse)
async def chat_ui(request: Request):
    """Render the chat interface."""
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "title": "Travel Planning Assistant"
        }
    )

@app.post("/api/chat")
async def chat(message: str):
    """Handle chat messages."""
    # This is a placeholder - in a real app, you'd process the message with your AI
    response = {"response": f"I received your message: {message}"}
    return JSONResponse(content=response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
