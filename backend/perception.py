import os
import time

def transcribe_audio(file_path: str) -> str:
    """
    Transcribe audio files into text witness statements.
    Simulates transcription processing with realistic mock transcripts based on filename
    to avoid heavy model downloads and slow runtime on standard CPUs.
    """
    filename = os.path.basename(file_path).lower()
    
    # Simulate a brief processing delay
    time.sleep(1.0)
    
    # Context-specific high-quality transcripts
    if "john" in filename or "guard" in filename:
        return (
            "Official Statement Transcript (Security Guard John): "
            "I was stationed at the main entrance desk. At 10:00 PM, the alarm sounded. "
            "I rushed to the scene and saw a tall figure in a black hoodie exiting toward the north gate. "
            "I did not observe anyone else, and I stayed at my post until police arrived at 10:15 PM."
        )
    elif "clara" in filename or "visitor" in filename:
        return (
            "Recorded Phone Testimony Transcript (Visitor Clara): "
            "I was waiting for a taxi outside the north exit around 9:55 PM. "
            "I saw a man in a red jacket sprinting out of the building. "
            "Also, earlier around 9:45 PM, I saw the security guard John whispering to someone "
            "behind the exhibition columns. They exchanged a bag."
        )
    
    # Generic fallback transcript
    return (
        f"Transcribed audio feed ({os.path.basename(file_path)}): "
        "A loud noise was heard near the gallery wing around 9:55 PM. "
        "A witness reports seeing a silver sedan driving away with its headlights turned off."
    )

def analyze_video(file_path: str):
    """
    Analyze video files (CCTV) and extract timestamped object/person tracking logs.
    Utilizes a real YOLOv8 model if ultralytics is installed, otherwise falls back to simulated logs.
    """
    try:
        from ultralytics import YOLO
        import cv2
        has_yolo = True
    except ImportError:
        has_yolo = False
        
    if has_yolo:
        try:
            print(f"Loading YOLOv8 Nano model to analyze {file_path}...")
            # Automatically downloads yolov8n.pt (~6.2MB) to current directory if not present
            model = YOLO("yolov8n.pt") 
            cap = cv2.VideoCapture(file_path)
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logs = []
            frame_idx = 0
            
            # Sample 1 frame per second to optimize speed on standard CPUs
            sample_rate = int(fps)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_idx % sample_rate == 0:
                    current_seconds = frame_idx / fps
                    timestamp_str = time.strftime('%H:%M:%S', time.gmtime(current_seconds))
                    
                    # Run inference on the single sampled frame
                    results = model(frame, verbose=False)
                    for r in results:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            label = model.names[cls_id]
                            conf = float(box.conf[0])
                            
                            # Filter for entities relevant to investigations (people, bags, vehicles)
                            if label in ["person", "car", "truck", "motorcycle", "bicycle", "backpack", "handbag", "suitcase"]:
                                logs.append({
                                    "timestamp": f"Video Time +{timestamp_str}",
                                    "object": f"{label.capitalize()}",
                                    "location": "CCTV Frame",
                                    "confidence": conf
                                })
                frame_idx += 1
                
            cap.release()
            if logs:
                print(f"YOLOv8 analysis completed. Extracted {len(logs)} frames entries.")
                return logs
        except Exception as e:
            print(f"YOLOv8 execution failed, falling back to mock logs: {e}")
            
    # Falls back to simulated logs if libraries are missing or execution fails
    filename = os.path.basename(file_path).lower()
    time.sleep(1.5)
    
    if "cctv" in filename or "camera" in filename:
        return [
            {"timestamp": "09:45:10 PM", "object": "Person (Guard John & Suspect)", "location": "Gallery Columns", "confidence": 0.94},
            {"timestamp": "09:53:00 PM", "object": "Object (Glass Case Shattered)", "location": "Display Hall", "confidence": 0.98},
            {"timestamp": "09:58:15 PM", "object": "Person (Hooded Figure)", "location": "North Exit Gate", "confidence": 0.91},
            {"timestamp": "09:58:30 PM", "object": "Vehicle (Escape Car)", "location": "North Alleyway", "confidence": 0.85}
        ]
    
    # Default visual logs
    return [
        {"timestamp": "10:00:05 PM", "object": "Person (Suspect)", "location": "Zone A", "confidence": 0.89},
        {"timestamp": "10:02:15 PM", "object": "Vehicle (Escape Car)", "location": "Zone B", "confidence": 0.82}
    ]
