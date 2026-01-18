FROM python:3.11-slim

# Install dependencies
RUN pip install --no-cache-dir \
    requests \
    numpy \
    pandas \
    regex \
    fastapi \
    uvicorn[standard] \
    python-dotenv

# Create directories
WORKDIR /app
RUN mkdir -p /mnt/data

# Copy the RLM package (needed for recursive agent spawning)
COPY rlm/ ./rlm/

# Copy the server
COPY repl_server.py .

# Expose HTTP port
EXPOSE 8080

# Environment defaults
ENV PORT=8080
ENV HOST=0.0.0.0
ENV RLM_MAX_RECURSION_DEPTH=3
ENV PYTHONPATH=/app

# Run the HTTP server
CMD ["python", "-u", "repl_server.py"]
