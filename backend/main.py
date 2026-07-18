from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

# Load environment variables (e.g., GEMINI_API_KEY)
load_dotenv()

app = FastAPI(
    title="Multi-Agent Investigation Intelligence System API",
    description="Backend API for the autonomous crime investigation system.",
    version="1.0.0"
)

# In-memory storage for MVP (will be replaced by DB later)
investigation_state = {
    "evidence": [],
    "timeline": [],
    "suspects": [],
    "contradictions": [],
    "status": "idle"
}

from backend.agents.orchestrator import DetectiveAgent
from backend.db import db_manager
from backend.perception import transcribe_audio, analyze_video

detective_agent = DetectiveAgent()

class EvidenceText(BaseModel):
    source: str # e.g., "Witness 1", "Police Report"
    content: str
    timestamp: Optional[str] = None

@app.get("/")
def read_root():
    return {"message": "MAIIS API is running."}

@app.post("/api/evidence/text")
def add_text_evidence(evidence: EvidenceText):
    """Upload witness statements or text reports."""
    investigation_state["evidence"].append({
        "type": "text",
        "source": evidence.source,
        "content": evidence.content,
        "timestamp": evidence.timestamp
    })
    return {"message": "Text evidence added successfully."}

@app.post("/api/evidence/file")
async def add_file_evidence(source: str = Form(...), file: UploadFile = File(...)):
    """Upload images, video, or audio files."""
    # Ensure upload directory exists
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"
    
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
        
    investigation_state["evidence"].append({
        "type": "file",
        "source": source,
        "filename": file.filename,
        "filepath": file_path
    })
    
    # Trigger transcription or visual tracking logs dynamically based on file type
    lower_filename = file.filename.lower()
    if lower_filename.endswith((".wav", ".mp3", ".m4a")):
        transcript = transcribe_audio(file_path)
        investigation_state["evidence"].append({
            "type": "text",
            "source": f"{source} (Audio Transcript)",
            "content": transcript
        })
    elif lower_filename.endswith((".mp4", ".avi", ".mov")):
        logs = analyze_video(file_path)
        investigation_state["evidence"].append({
            "type": "video_logs",
            "source": f"{source} (CCTV Tracker)",
            "content": f"Automated CV Tracking Completed: {len(logs)} entities recorded.",
            "logs": logs
        })
        
    return {"message": f"File {file.filename} uploaded and processed successfully."}

@app.post("/api/investigate")
def run_investigation():
    """Trigger the multi-agent investigation loop."""
    global investigation_state
    investigation_state["status"] = "running"
    
    # Call the Detective Agent to orchestrate the sub-agents
    investigation_state = detective_agent.investigate(investigation_state)
    
    # Seed standard graph schema nodes
    db_manager.add_entity("golden_falcon", "Golden Falcon Statue", "Stolen Object")
    db_manager.add_entity("gallery", "Museum Display Hall", "Location")
    db_manager.add_entity("north_exit", "Museum North Exit", "Location")
    db_manager.add_relation("gallery", "golden_falcon", "LOCATED_AT")
    
    # Extract entities and build relations dynamically from AI outputs
    for suspect in investigation_state.get("suspects", []):
        reason_text = suspect.get("reason", "").lower()
        if "john" in reason_text:
            db_manager.add_entity("john", "Security Guard John", "Suspect")
            db_manager.add_relation("john", "golden_falcon", "IMPLICATED_IN")
            db_manager.add_relation("john", "gallery", "GUARDED")
        if "clara" in reason_text:
            db_manager.add_entity("clara", "Visitor Clara", "Witness")
            db_manager.add_relation("clara", "north_exit", "LOCATED_AT")
        if "red jacket" in reason_text:
            db_manager.add_entity("suspect_red_jacket", "Man in Red Jacket", "Suspect")
            db_manager.add_relation("suspect_red_jacket", "north_exit", "SEEN_AT")
            db_manager.add_relation("john", "suspect_red_jacket", "COLLUDED_WITH")
            
    return {"message": "Investigation completed.", "results": investigation_state}

@app.get("/api/status")
def get_status():
    """Get the current state of the investigation."""
    return investigation_state

@app.get("/api/graph")
def get_graph():
    """Get the NetworkX graph node-link relation data."""
    return db_manager.get_graph_data()

@app.post("/api/reset")
def reset_investigation():
    """Reset the investigation state."""
    global investigation_state
    investigation_state = {
        "evidence": [],
        "timeline": [],
        "suspects": [],
        "contradictions": [],
        "status": "idle"
    }
    db_manager.clear()
    return {"message": "Investigation state reset successfully."}

