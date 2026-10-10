# nnviz — interactive 3D neural network explorer (CPU image)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NNVIZ_HOST=0.0.0.0 \
    NNVIZ_PORT=8765 \
    NNVIZ_OPEN_BROWSER=0 \
    NNVIZ_UPLOAD_DIR=/data/uploads \
    TORCH_HOME=/data/torch

# unprivileged user; /data holds uploads and the pretrained-weights cache (mount it as a volume)
RUN useradd --create-home --uid 10001 nnviz \
 && mkdir -p /data/uploads /data/torch \
 && chown -R nnviz:nnviz /data

WORKDIR /app

# CPU-only PyTorch keeps the image small (~1 GB instead of several GB with CUDA wheels)
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install -r requirements.txt

COPY --chown=nnviz:nnviz . .

USER nnviz
VOLUME ["/data"]
EXPOSE 8765

# the toy model trains for a few seconds at start-up, hence the generous start period
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/' % os.environ.get('NNVIZ_PORT','8765'), timeout=4)"

CMD ["python", "server.py"]
