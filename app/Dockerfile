FROM python:3.11-slim

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# server.py, static/ and model/ (model.onnx + meta.json)
COPY . .

# Hugging Face Spaces runs containers as uid 1000 and expects port 7860
RUN useradd -m -u 1000 user && chown -R user /srv
USER user
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
