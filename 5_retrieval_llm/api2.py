# this is the alternative api endpoints update from the api.py 

import uuid
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from main import query_supabase, call_openai_llm
from typing import List, Dict, Any, Optional
import uvicorn

app = FastAPI()

# In-memory storage for chat histories (in production, you might use a database)
chat_histories = {}

def generate_session_id():
    return str(uuid.uuid4())

class QueryRequest(BaseModel):
    user_query: str
    chat_history: List[Dict[str, str]] = []
    session_id: Optional[str] = None

class ChatHistoryRequest(BaseModel):
    session_id: str = "default"

# Convert anything that FastAPI can't serialize
def sanitize(obj):
    if isinstance(obj, np.generic):  # np.float32, np.int64, etc.
        return obj.item()
    elif isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize(i) for i in obj)
    return obj

@app.get("/")
def root():
    return {"message": "Document Retrieval and LLM API is running."}

@app.post("/query")
def query_documents(request: QueryRequest):
    try:
        retrieved_chunks = query_supabase(request.user_query)
        sanitized_chunks = sanitize(retrieved_chunks)
        return {"retrieved_chunks": sanitized_chunks}
    except Exception as e:
        print("Error in /query:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat_with_llm(request: QueryRequest, return_history: bool = False):
    try:
        # Get existing chat history for this session
        session_id = request.session_id or generate_session_id()

        existing_history = chat_histories.get(session_id, [])
        
        # Handle special commands
        if request.user_query.lower() == "new chat":
            session_id = generate_session_id()
            chat_histories[session_id] = []
            return {
                "answer": "Starting a new chat...\n\nHow can I help you today?"
            } if not return_history else {
                "answer": "Starting a new chat...\n\nHow can I help you today?",
                "chat_history": [],
                "session_id": session_id
            }
            
        if request.user_query.lower() == "show history":
            history_text = "Chat History:\n\n"
            for msg in existing_history:
                history_text += f"{msg['role']}: {msg['content']}\n\n"
            return {
                "answer": history_text
            } if not return_history else {
                "answer": history_text,
                "chat_history": existing_history,
                "session_id": session_id
            }
            
        if request.user_query.lower() == "exit":
            return {
                "answer": "Chat ended. Please close this window or start a new chat."
            } if not return_history else {
                "answer": "Chat ended. Please close this window or start a new chat.",
                "chat_history": existing_history,
                "session_id": session_id
            }
        
        # Use the existing history if provided in the request
        history_to_use = request.chat_history if request.chat_history else existing_history
        
        # Retrieve document chunks for context
        retrieved_chunks = query_supabase(request.user_query)
        
        # Get response from LLM
        answer, updated_history = call_openai_llm(
            request.user_query, retrieved_chunks, history_to_use
        )
        
        # Update stored chat history
        chat_histories[session_id] = updated_history
        
        # Return just the answer or include history based on parameter
        return {
            "answer": answer,
            "session_id": session_id
        } if not return_history else {
            "answer": answer, 
            "chat_history": updated_history,
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{session_id}")
def get_chat_history(session_id: str = "default"):
    """Endpoint to get chat history for a specific session."""
    history = chat_histories.get(session_id, [])
    return {"chat_history": history, "session_id": session_id}

@app.delete("/history/{session_id}")
def clear_chat_history(session_id: str = "default"):
    """Endpoint to clear chat history for a specific session."""
    chat_histories[session_id] = []
    return {"status": "success", "message": "Chat history cleared", "session_id": session_id}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8500)
