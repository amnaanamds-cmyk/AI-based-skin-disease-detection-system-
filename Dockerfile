FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
# CPU wheels keep the image small; swap the index URL for a CUDA build if serving on GPU.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install -r requirements.txt

COPY dermaai ./dermaai
COPY web ./web

RUN useradd --create-home app
USER app

# Mount trained weights at /models/dermaai.pt; without them the API runs in demo mode.
ENV DERMAAI_CHECKPOINT=/models/dermaai.pt
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "dermaai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
