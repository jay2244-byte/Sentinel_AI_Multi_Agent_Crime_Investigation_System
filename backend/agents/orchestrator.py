from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from .sub_agents import EvidenceAnalyzer, TimelineBuilder, ContradictionDetector, SuspectReasoner

class AgentState(TypedDict):
    evidence: List[Dict[str, Any]]
    analysis: str
    timeline: List[str]
    contradictions: List[str]
    suspects: List[Dict[str, Any]]
    status: str

class DetectiveAgent:
    """The central orchestrator of the MAIIS utilizing stateful LangGraph loops."""
    
    def __init__(self):
        self.evidence_analyzer = EvidenceAnalyzer()
        self.timeline_builder = TimelineBuilder()
        self.contradiction_detector = ContradictionDetector()
        self.suspect_reasoner = SuspectReasoner()
        self.workflow = self._compile_graph()
        
    def _analyze_evidence_node(self, state: AgentState) -> Dict[str, Any]:
        print("[Agent Node] Analyzing evidence...")
        analysis = self.evidence_analyzer.analyze(state["evidence"])
        return {"analysis": analysis}
        
    def _build_timeline_node(self, state: AgentState) -> Dict[str, Any]:
        print("[Agent Node] Building timeline...")
        timeline = self.timeline_builder.build_timeline(state["evidence"])
        return {"timeline": timeline}
        
    def _detect_contradictions_node(self, state: AgentState) -> Dict[str, Any]:
        print("[Agent Node] Detecting contradictions...")
        contradictions = self.contradiction_detector.detect(state["evidence"])
        return {"contradictions": contradictions}
        
    def _reason_suspects_node(self, state: AgentState) -> Dict[str, Any]:
        print("[Agent Node] Reasoning about suspects...")
        timeline = state["timeline"]
        contradictions = state["contradictions"]
        evidence = state["evidence"]
        suspects = self.suspect_reasoner.deduce_suspects(evidence, timeline, contradictions)
        return {"suspects": suspects, "status": "completed"}

    def _compile_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        # Add Nodes
        workflow.add_node("analyze_evidence", self._analyze_evidence_node)
        workflow.add_node("build_timeline", self._build_timeline_node)
        workflow.add_node("detect_contradictions", self._detect_contradictions_node)
        workflow.add_node("reason_suspects", self._reason_suspects_node)
        
        # Define entry point
        workflow.set_entry_point("analyze_evidence")
        
        # Branch out in parallel
        workflow.add_edge("analyze_evidence", "build_timeline")
        workflow.add_edge("analyze_evidence", "detect_contradictions")
        
        # Merge parallel results back
        workflow.add_edge("build_timeline", "reason_suspects")
        workflow.add_edge("detect_contradictions", "reason_suspects")
        
        # Complete workflow
        workflow.add_edge("reason_suspects", END)
        
        return workflow.compile()

    def investigate(self, global_state: dict) -> dict:
        print("Detective Agent starting LangGraph flow...")
        initial_state: AgentState = {
            "evidence": global_state.get("evidence", []),
            "analysis": "",
            "timeline": [],
            "contradictions": [],
            "suspects": [],
            "status": "running"
        }
        
        # Run graph execution
        result_state = self.workflow.invoke(initial_state)
        
        # Map output back to global state dict
        global_state["timeline"] = result_state["timeline"]
        global_state["contradictions"] = result_state["contradictions"]
        global_state["suspects"] = result_state["suspects"]
        global_state["status"] = "completed"
        return global_state
