import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Force reloading of env variables
load_dotenv(override=True)

# Initialize the LLM (Groq, local Ollama, or Gemini)
def get_llm():
    # 1. Local Ollama Fallback
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
    if use_ollama:
        from langchain_ollama import ChatOllama
        model_name = os.getenv("OLLAMA_MODEL", "llama3")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        print(f"[LLM Router] Routing to local Ollama ({model_name}) at {base_url}...")
        return ChatOllama(model=model_name, base_url=base_url, temperature=0.1)
        
    # 2. Google Gemini Fallback
    google_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    use_gemini = os.getenv("USE_GEMINI", "false").lower() == "true"
    if use_gemini and google_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("[LLM Router] Routing to Google Gemini (gemini-1.5-flash)...")
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, google_api_key=google_key)
        
    # 3. Default Groq Client
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.startswith("gsk_placeholder") or "YOUR_GROQ_API_KEY" in api_key:
        # If Groq key is missing, check if Gemini key is available as automatic fallback
        if google_key:
            from langchain_google_genai import ChatGoogleGenerativeAI
            print("[LLM Router] Groq key missing. Falling back to Google Gemini...")
            return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, google_api_key=google_key)
        
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, api_key="PLACEHOLDER_KEY")
        
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)



class EvidenceAnalyzer:
    def __init__(self):
        self.name = "Evidence Analyzer"
        self.prompt = PromptTemplate(
            input_variables=["evidence_text"],
            template="You are an expert police Evidence Analyzer.\n"
                     "Extract all entities (People, Locations, Objects, Vehicles) from the evidence below.\n"
                     "Return a clear, concise bulleted list.\n\n"
                     "Evidence:\n{evidence_text}"
        )
        self.chain = self.prompt | get_llm() | StrOutputParser()

    def analyze(self, evidence_data):
        texts = [f"Source ({e['source']}): {e['content']}" for e in evidence_data if e["type"] == "text"]
        if not texts:
            return "No text evidence available for analysis."
        return self.chain.invoke({"evidence_text": "\n\n".join(texts)})

class TimelineBuilder:
    def __init__(self):
        self.name = "Timeline Builder"
        self.prompt = PromptTemplate(
            input_variables=["evidence_text"],
            template="You are an expert Timeline Builder for police investigations.\n"
                     "Extract all events and timestamps from the evidence below and arrange them chronologically.\n"
                     "Return only a bulleted list of events with their estimated times.\n\n"
                     "Evidence:\n{evidence_text}"
        )
        self.chain = self.prompt | get_llm() | StrOutputParser()

    def build_timeline(self, evidence_data):
        texts = [f"Source ({e['source']}): {e['content']}" for e in evidence_data if e["type"] == "text"]
        if not texts:
            return ["No timeline can be established without text evidence."]
        result = self.chain.invoke({"evidence_text": "\n\n".join(texts)})
        # Split into list for the frontend
        return [item.strip() for item in result.split('\n') if item.strip()]

class ContradictionDetector:
    def __init__(self):
        self.name = "Contradiction Detector"
        self.prompt = PromptTemplate(
            input_variables=["evidence_text"],
            template="You are a Contradiction Detector.\n"
                     "Compare the differing testimonies and evidence provided below.\n"
                     "Highlight any logical flaws, direct contradictions, or conflicting statements.\n"
                     "Return a bulleted list of specific contradictions.\n\n"
                     "Evidence:\n{evidence_text}"
        )
        self.chain = self.prompt | get_llm() | StrOutputParser()

    def detect(self, evidence_data):
        texts = [f"Source ({e['source']}): {e['content']}" for e in evidence_data if e["type"] == "text"]
        if not texts:
            return ["No evidence to compare."]
        result = self.chain.invoke({"evidence_text": "\n\n".join(texts)})
        return [item.strip() for item in result.split('\n') if item.strip()]

class SuspectReasoner:
    def __init__(self):
        self.name = "Suspect Reasoner"
        self.prompt = PromptTemplate(
            input_variables=["evidence_text", "timeline_text", "contradictions_text"],
            template="You are the Lead Suspect Reasoner.\n"
                     "Using the raw evidence, the established timeline, and noted contradictions, deduce the most likely suspects.\n"
                     "Analyze motive, means, and opportunity.\n"
                     "Return a concise list of suspects and a 1-sentence reason for each.\n\n"
                     "Timeline:\n{timeline_text}\n\n"
                     "Contradictions:\n{contradictions_text}\n\n"
                     "Evidence:\n{evidence_text}"
        )
        self.chain = self.prompt | get_llm() | StrOutputParser()

    def deduce_suspects(self, evidence_data, timeline, contradictions):
        texts = [f"Source ({e['source']}): {e['content']}" for e in evidence_data if e["type"] == "text"]
        if not texts:
            return [{"name": "Unknown", "reason": "No evidence provided."}]
            
        timeline_str = "\n".join(timeline) if isinstance(timeline, list) else timeline
        contradictions_str = "\n".join(contradictions) if isinstance(contradictions, list) else contradictions
        
        result = self.chain.invoke({
            "evidence_text": "\n\n".join(texts),
            "timeline_text": timeline_str,
            "contradictions_text": contradictions_str
        })
        
        # We need to parse the string result into a dict for the frontend.
        # Let's just return a generic dictionary with the LLM text output for the MVP.
        return [{"name": "AI Output", "reason": result}]
