# ==============================================================================
# Dockerfile: Image Captioning API Service (CNN + LSTM)
# ------------------------------------------------------------------------------
# Builds a lightweight Python 3.11 container to serve the Flickr8k image
# captioning model using FastAPI and Uvicorn. CPU-only PyTorch is installed
# to minimize final image size.
# ==============================================================================

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for optimal Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy application code and trained model artifacts
COPY app/ ./app/
COPY src/ ./src/
COPY settings.py .
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]