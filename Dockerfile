FROM python:3.11-slim

# Install dependencies
RUN pip install --no-cache-dir \
    requests \
    numpy \
    pandas \
    regex

# Create working directory
WORKDIR /app

# Copy the REPL server
COPY repl_server.py .

# Run the REPL server (unbuffered output)
CMD ["python", "-u", "repl_server.py"]
