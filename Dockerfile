FROM python:3.12-slim

WORKDIR /app

COPY phase1/requirements.in ./phase1/requirements.in
RUN pip install --no-cache-dir -r phase1/requirements.in

COPY phase1/app.py ./phase1/app.py
COPY phase1/__init__.py ./phase1/__init__.py

ENV PORT=8002

EXPOSE 8002

HEALTHCHECK --interval=5s --timeout=3s --start-period=10s --retries=6 \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8002\")}/health',timeout=2).status==200 else 1)"

CMD ["sh", "-c", "uvicorn phase1.app:app --host 0.0.0.0 --port $PORT"]
