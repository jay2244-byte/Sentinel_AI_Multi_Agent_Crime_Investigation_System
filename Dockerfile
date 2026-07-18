FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and YOLO
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install extra required dependencies for video perception & local LLM
RUN pip install --no-cache-dir langchain-ollama ultralytics

# Copy the rest of the application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["python", "-m", "streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
