import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import os
import sys
from dotenv import load_dotenv

# Load/reload .env environment variables dynamically
load_dotenv(override=True)

# Add project root directory to sys.path so we can import 'backend'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.agents.orchestrator import DetectiveAgent
from backend.db import db_manager
from backend.perception import transcribe_audio, analyze_video

# Initialize in-process investigation state in Streamlit session state
if "investigation_state" not in st.session_state:
    st.session_state.investigation_state = {
        "evidence": [],
        "timeline": [],
        "suspects": [],
        "contradictions": [],
        "status": "idle"
    }

def draw_network_graph(graph_data):
    if not graph_data or not graph_data.get("nodes"):
        st.info("No entity links mapped yet. Run investigation first.")
        return
        
    G = nx.DiGraph()
    for node in graph_data["nodes"]:
        G.add_node(node["id"], label=node["label"], type=node["type"])
    for edge in graph_data["edges"]:
        G.add_edge(edge["source"], edge["target"], label=edge["label"])
        
    # Configure Matplotlib styling to fit standard Streamlit themes
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    fig.patch.set_facecolor('#0e1117') # Match Streamlit dark page background
    ax.set_facecolor('#0e1117')
    
    pos = nx.spring_layout(G, k=1.0, seed=42)
    
    # Custom color mapping for node types
    node_colors = []
    for node in G.nodes():
        ntype = G.nodes[node].get("type", "")
        if ntype == "Suspect":
            node_colors.append("#ff4b4b")  # Streamlit Primary Red
        elif ntype == "Witness":
            node_colors.append("#ffaa00")  # Orange
        elif ntype == "Stolen Object":
            node_colors.append("#29b5e8")  # Blue
        elif ntype == "Location":
            node_colors.append("#00cc96")  # Green
        else:
            node_colors.append("#a3a7b2")  # Grey fallback
            
    # Draw Nodes and Labels
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1800, alpha=0.9, ax=ax)
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color="#ffffff", font_weight="bold", ax=ax)
    
    # Draw Edges with Arrows
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color="#636c7c", width=1.5, arrowstyle="->", arrowsize=15, min_source_margin=15, min_target_margin=15, ax=ax)
    
    # Draw Edge Labels (Relations)
    edge_labels = nx.get_edge_attributes(G, 'type')
    if not edge_labels:
        edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color="#a3a7b2", rotate=False, ax=ax)
    
    st.pyplot(fig)

st.set_page_config(page_title="MAIIS Dashboard", layout="wide")

st.title("🕵️‍♂️ Multi-Agent Investigation Intelligence System (MAIIS)")
st.markdown("Upload evidence, trigger the AI agents, and view the investigation results.")

# Sidebar for evidence upload
st.sidebar.header("Upload Evidence")

# Case Loaders & Reset
if st.sidebar.button("📂 Load Mock Case (Golden Falcon)", use_container_width=True):
    # Reset
    st.session_state.investigation_state = {
        "evidence": [],
        "timeline": [],
        "suspects": [],
        "contradictions": [],
        "status": "idle"
    }
    db_manager.clear()
    
    # Load samples
    samples = [
        {"source": "Security Guard John", "content": "I was on duty at the main gallery entrance. Around 10:00 PM, the security alarm went off. I immediately ran to the display hall and saw a suspect wearing a black hoodie running out of the gallery towards the north exit. I did not see anyone else around, and I stayed at my post until the police arrived at 10:15 PM."},
        {"source": "Visitor Clara", "content": "I was outside the museum near the north exit around 9:55 PM waiting for a taxi. I saw a man wearing a red jacket run out of the building. Just before that, at around 9:45 PM, I noticed the security guard John talking quietly to someone behind the gallery pillars. They seemed to be exchanging a small bag."},
        {"source": "Forensic Report", "content": "The Golden Falcon was stolen from its display case on June 5th. The display case glass was broken, triggering the silent alarm at 9:53 PM. Security camera footage shows a figure in a black hoodie exiting the gallery at 9:58 PM. Guard John did not log or report the alarm manually until 10:05 PM."}
    ]
    for s in samples:
        st.session_state.investigation_state["evidence"].append({
            "type": "text",
            "source": s["source"],
            "content": s["content"],
            "timestamp": None
        })
        
    os.makedirs("uploads", exist_ok=True)
    video_path = "uploads/cctv_north_gate.mp4"
    with open(video_path, "wb") as f:
        f.write(b"dummy_mp4_bytes")
        
    logs = analyze_video(video_path)
    st.session_state.investigation_state["evidence"].append({
        "type": "video_logs",
        "source": "North Gate CCTV Camera",
        "content": f"Automated CV Tracking Completed: {len(logs)} entities recorded.",
        "logs": logs
    })
    st.sidebar.success("Loaded 'Golden Falcon' case with CCTV video!")
    st.rerun()

if st.sidebar.button("📹 Load Real CCTV Case (Person/Car)", use_container_width=True):
    # Reset
    st.session_state.investigation_state = {
        "evidence": [],
        "timeline": [],
        "suspects": [],
        "contradictions": [],
        "status": "idle"
    }
    db_manager.clear()
    
    # Check if download is needed
    video_path = "uploads/person_bicycle_car_detection.mp4"
    if not os.path.exists(video_path):
        with st.spinner("Downloading real test CCTV video (~1MB)..."):
            import subprocess
            subprocess.run(["python", "extra/download_realtime_dataset.py"])
            
    # Load witness statements
    stmt_a_path = "uploads/statement_transit_point_a.txt"
    stmt_b_path = "uploads/statement_patrol_officer.txt"
    
    stmt_a = "Witness Statement (Transit Point A):\nI was walking near the main crossing around 10:05 PM. I noticed a person riding a bicycle. A few moments later, a car drove past very quickly."
    stmt_b = "Transit Patrol Log (Officer Davis):\nAt 10:10 PM, a car was observed traveling above the speed limit. A person on a bicycle was seen waiting at the intersection, seemingly watching the car's movement."
    
    if os.path.exists(stmt_a_path):
        with open(stmt_a_path, "r") as f:
            stmt_a = f.read()
    if os.path.exists(stmt_b_path):
        with open(stmt_b_path, "r") as f:
            stmt_b = f.read()
            
    st.session_state.investigation_state["evidence"].append({
        "type": "text",
        "source": "Witness Statement (Transit Point A)",
        "content": stmt_a,
        "timestamp": "10:05 PM"
    })
    st.session_state.investigation_state["evidence"].append({
        "type": "text",
        "source": "Patrol Officer Davis Log",
        "content": stmt_b,
        "timestamp": "10:10 PM"
    })
    
    # Process real video with YOLOv8
    with st.spinner("YOLOv8 is analyzing real-world footage frame-by-frame..."):
        logs = analyze_video(video_path)
        st.session_state.investigation_state["evidence"].append({
            "type": "video_logs",
            "source": "Transit CCTV Camera",
            "content": f"YOLOv8 Analysis Completed: {len(logs)} entities detected in raw footage.",
            "logs": logs
        })
        
    st.sidebar.success("Loaded Real CCTV Video case!")
    st.rerun()

if st.sidebar.button("🛍️ Load Robbery CCTV Case (Store Aisle)", use_container_width=True):
    # Reset
    st.session_state.investigation_state = {
        "evidence": [],
        "timeline": [],
        "suspects": [],
        "contradictions": [],
        "status": "idle"
    }
    db_manager.clear()
    
    # Check if download is needed
    video_path = "uploads/robbery_store_aisle.mp4"
    if not os.path.exists(video_path):
        with st.spinner("Downloading store robbery video (~1MB)..."):
            import subprocess
            subprocess.run(["python", "extra/download_robbery_dataset.py"])
            
    # Load witness statements
    stmt_a = "Witness Statement 1 (Store Clerk - Sarah):\nI was working at the cash register around 9:45 PM. A tall man wearing a dark jacket and sunglasses walked into the store. He walked down the main store aisle, pulled out a small black bag, and demanded all the cash in the register. He took approximately $500 and ran out towards the rear alleyway."
    stmt_b = "Witness Statement 2 (Customer - Michael):\nI was in the store aisle looking at snacks around 9:43 PM. I saw a man walking in a suspicious way, looking around. He had a black backpack. He walked to the register and confronted the clerk. He seemed to have something in his jacket pocket, although I did not see a weapon. I immediately hid behind the columns."
    stmt_c = "Witness Statement 3 (Passerby - David):\nI was walking past the store front around 9:48 PM. I saw a man in a dark jacket with a black backpack running out of the store. He ran into the rear alleyway. Just before that, around 9:40 PM, I noticed a silver sedan idling in the alleyway with its license plate covered."
    
    st.session_state.investigation_state["evidence"].append({
        "type": "text",
        "source": "Store Clerk Sarah",
        "content": stmt_a,
        "timestamp": "9:45 PM"
    })
    st.session_state.investigation_state["evidence"].append({
        "type": "text",
        "source": "Customer Michael",
        "content": stmt_b,
        "timestamp": "9:43 PM"
    })
    st.session_state.investigation_state["evidence"].append({
        "type": "text",
        "source": "Passerby David",
        "content": stmt_c,
        "timestamp": "9:48 PM"
    })
    
    # Process real video with YOLOv8
    with st.spinner("YOLOv8 is analyzing store CCTV footage frame-by-frame..."):
        logs = analyze_video(video_path)
        st.session_state.investigation_state["evidence"].append({
            "type": "video_logs",
            "source": "Store Aisle CCTV Camera",
            "content": f"YOLOv8 Analysis Completed: {len(logs)} entities detected in raw footage.",
            "logs": logs
        })
        
    st.sidebar.success("Loaded Convenience Store Robbery Case!")
    st.rerun()

if st.sidebar.button("🗑️ Clear All Evidence", use_container_width=True):
    st.session_state.investigation_state = {
        "evidence": [],
        "timeline": [],
        "suspects": [],
        "contradictions": [],
        "status": "idle"
    }
    db_manager.clear()
    st.sidebar.success("Cleared all evidence.")
    st.rerun()

# Text Evidence
if "witness_count" not in st.session_state:
    st.session_state.witness_count = 1

with st.sidebar.expander("📝 Add Text Evidence", expanded=True):
    witnesses = []
    for i in range(st.session_state.witness_count):
        st.markdown(f"**Witness Entry {i+1}**")
        src = st.text_input(f"Source (e.g., Witness {i+1})", key=f"src_{i}")
        content = st.text_area(f"Statement/Content", key=f"content_{i}")
        witnesses.append((src, content))
        st.divider()
        
    col_add, col_sub = st.columns(2)
    with col_add:
        if st.button("➕ Add Field", use_container_width=True):
            st.session_state.witness_count += 1
            st.rerun()
    with col_sub:
        if st.button("💾 Submit All", use_container_width=True, type="primary"):
            success_count = 0
            for src, content in witnesses:
                if src.strip() and content.strip():
                    st.session_state.investigation_state["evidence"].append({
                        "type": "text",
                        "source": src.strip(),
                        "content": content.strip(),
                        "timestamp": None
                    })
                    success_count += 1
            if success_count > 0:
                st.sidebar.success(f"Added {success_count} statements!")
                st.session_state.witness_count = 1
                for key in list(st.session_state.keys()):
                    if key.startswith("src_") or key.startswith("content_"):
                        del st.session_state[key]
                st.rerun()
            else:
                st.sidebar.error("Fill in at least one statement.")

# File Evidence
with st.sidebar.expander("📹 Upload Media (CCTV/Audio)"):
    file_source = st.text_input("Media Source (e.g., CCTV Cam 4)")
    uploaded_file = st.file_uploader("Choose file")
    if st.button("Upload File"):
        if uploaded_file and file_source:
            os.makedirs("uploads", exist_ok=True)
            file_path = f"uploads/{uploaded_file.name}"
            with open(file_path, "wb") as buffer:
                buffer.write(uploaded_file.getvalue())
                
            st.session_state.investigation_state["evidence"].append({
                "type": "file",
                "source": file_source,
                "filename": uploaded_file.name,
                "filepath": file_path
            })
            
            # Run perception pipeline directly
            lower_filename = uploaded_file.name.lower()
            if lower_filename.endswith((".wav", ".mp3", ".m4a")):
                transcript = transcribe_audio(file_path)
                st.session_state.investigation_state["evidence"].append({
                    "type": "text",
                    "source": f"{file_source} (Audio Transcript)",
                    "content": transcript
                })
            elif lower_filename.endswith((".mp4", ".avi", ".mov")):
                logs = analyze_video(file_path)
                st.session_state.investigation_state["evidence"].append({
                    "type": "video_logs",
                    "source": f"{file_source} (CCTV Tracker)",
                    "content": f"Automated CV Tracking Completed: {len(logs)} entities recorded.",
                    "logs": logs
                })
                
            st.sidebar.success(f"Uploaded and processed {uploaded_file.name}!")
            st.rerun()

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Current Evidence")
    state = st.session_state.investigation_state
    if not state["evidence"]:
        st.info("No evidence uploaded yet.")
    for ev in state["evidence"]:
        if ev["type"] == "text":
            st.markdown(f"📝 **{ev['source']}**:")
            st.write(ev['content'])
        elif ev["type"] == "video_logs":
            with st.expander(f"📹 {ev['source']} (Visual Tracking Logs)"):
                for log in ev.get("logs", []):
                    st.caption(f"⏱️ **{log['timestamp']}** | {log['object']} seen at *{log['location']}* (Conf: {log['confidence']:.2f})")
        else:
            st.write(f"📁 **{ev['source']}** (File): {ev['filename']}")

with col2:
    st.subheader("Agent Actions")
    if st.button("🚀 Run Investigation", use_container_width=True):
        with st.spinner("Detective Agent is orchestrating sub-agents..."):
            try:
                st.session_state.investigation_state["status"] = "running"
                
                detective_agent = DetectiveAgent()
                st.session_state.investigation_state = detective_agent.investigate(st.session_state.investigation_state)
                
                # Seed standard graph schema nodes
                db_manager.add_entity("golden_falcon", "Golden Falcon Statue", "Stolen Object")
                db_manager.add_entity("gallery", "Museum Display Hall", "Location")
                db_manager.add_entity("north_exit", "Museum North Exit", "Location")
                db_manager.add_relation("gallery", "golden_falcon", "LOCATED_AT")
                
                # Extract entities and build relations dynamically from AI outputs
                for suspect in st.session_state.investigation_state.get("suspects", []):
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
                
                st.success("Investigation completed!")
                st.rerun()
            except Exception as e:
                st.session_state.investigation_state["status"] = "idle"
                err_str = str(e).lower()
                if "invalid_api_key" in err_str or "authentication" in err_str or "401" in err_str or "invalid api key" in err_str:
                    st.error("🔑 **API Key Error**: The LLM API Key is invalid or expired. Please check your `.env` file settings. You can also configure the Gemini or local Ollama fallback options if needed.")
                else:
                    st.error(f"❌ **Investigation failed**: {e}")

st.divider()

st.subheader("📊 Investigation Results")
state = st.session_state.investigation_state
if state["status"] == "completed":
    tab1, tab2 = st.tabs(["📋 Agent Analysis Reports", "🕸️ Interactive Knowledge Graph"])
    
    with tab1:
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("### 🕒 Timeline")
            for item in state.get("timeline", []):
                st.info(item)
        
        with c2:
            st.markdown("### ⚠️ Contradictions")
            for item in state.get("contradictions", []):
                st.warning(item)
        
        with c3:
            st.markdown("### 🕵️ Suspects")
            for item in state.get("suspects", []):
                st.error(f"**{item['name']}**: {item['reason']}")
                
        # Dossier Exporter
        st.markdown("---")
        st.subheader("📁 Case File Exporter")
        
        # Build report text
        dossier_text = f"# MAIIS CASE FILES: THE GOLDEN FALCON INVESTIGATION\n" \
                       f"Generated automatically by Multi-Agent Investigation Intelligence System\n\n" \
                       f"## 1. Timeline of Events\n"
        for item in state.get("timeline", []):
            dossier_text += f"* {item}\n"
            
        dossier_text += "\n## 2. Identified Contradictions & Discrepancies\n"
        for item in state.get("contradictions", []):
            dossier_text += f"* {item}\n"
            
        dossier_text += "\n## 3. Suspect Analysis & Lead Hypotheses\n"
        for item in state.get("suspects", []):
            dossier_text += f"* **{item['name']}**: {item['reason']}\n"
            
        st.download_button(
            label="📥 Download Complete Case Dossier (Markdown)",
            data=dossier_text,
            file_name="MAIIS_Case_Dossier.md",
            mime="text/markdown",
            use_container_width=True
        )
                
    with tab2:
        st.markdown("### 🕸️ Entity Relationship Mapping")
        graph_data = db_manager.get_graph_data()
        draw_network_graph(graph_data)
elif state["status"] == "idle":
    st.write("Waiting for investigation to start.")
