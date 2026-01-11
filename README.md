# RLM Engine - Recursive Language Model

An inference engine that enables LLMs to analyze documents and **codebases of any size** by treating them as external variables in a sandboxed Python environment. Based on the methodology from [Zhang et al. (MIT CSAIL)](https://arxiv.org/abs/2512.24601).

## Features

- **Unlimited Context** - Documents/directories loaded into sandbox memory, not LLM context window
- **Directory Analysis** - Analyze entire codebases with `list_files()`, `read_file()`, `search_files()`
- **Recursive Agents** - `llm_query()` spawns sub-agents for chunk analysis
- **Remote Deployment** - HTTP API for deploying as a remote service
- **Persistent State** - Variables persist across turns

## Quick Start

### 1. Setup

```bash
cd RLM
python -m venv env
source env/bin/activate
pip install -r requirements.txt
echo "OPENROUTER_API_KEY=your-key-here" > .env
```

### 2. Single File Analysis (Local Docker)

```bash
# Build the sandbox image
docker build -t rlm-sandbox .

# Analyze a single file
python main.py "Summarize the main themes" book.txt
python main.py "Find all function definitions" code.py --type "Python code"
```

### 3. Directory Analysis (Remote Server)

```bash
# Start the remote server with your codebase mounted
docker run -d --name rlm-server \
    -p 8080:8080 \
    -v /path/to/your/project:/mnt/data:ro \
    -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
    rlm-sandbox

# Analyze the directory from any machine
python main.py "Explain this codebase architecture" \
    --remote http://localhost:8080 \
    --directory /path/to/your/project
```

Or use docker-compose:
```bash
# Create data directory and copy your project
mkdir data && cp -r /path/to/your/project/* data/

# Start server
docker-compose up -d

# Query
python main.py "Find security issues" --remote http://localhost:8080 --directory ./data
```

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  User: "Explain this codebase"                              │
└─────────────────┬───────────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Root Agent (LLM)                                           │
│  → files = list_files("*.py")                               │
│  → content = read_file("main.py")                           │
│  → summary = llm_query(f"Summarize: {content}")             │
└─────────────────┬───────────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Remote Sandbox (Docker)                                    │
│  - HTTP API on port 8080                                    │
│  - Files indexed from /mnt/data                             │
│  - Persistent Python namespace                              │
└─────────────────────────────────────────────────────────────┘
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | Health check, file count |
| `/files` | GET | List indexed files |
| `/file/{path}` | GET | Read a specific file |
| `/execute` | POST | Execute Python code |
| `/reindex` | POST | Re-scan the data directory |
| `/reset` | POST | Clear the namespace |

## CLI Options

| Option | Description |
|--------|-------------|
| `--directory`, `-d` | Path to directory (enables directory mode) |
| `--remote`, `-r` | URL of remote sandbox server |
| `--remote-key` | API key for remote authentication |
| `--model`, `-m` | OpenRouter model (default: `xiaomi/mimo-v2-flash:free`) |
| `--max-turns` | Max conversation turns (default: 15) |
| `--type`, `-t` | Content type hint (e.g., "codebase", "book") |
| `--quiet`, `-q` | Suppress verbose output |

## Available Functions in Sandbox

```python
list_files(pattern="*")     # List files matching glob pattern
read_file(path)             # Read a file's content
search_files(regex, glob)   # Search across files
llm_query(prompt, model)    # Spawn sub-agent for analysis
get_file_tree()             # Get nested directory structure
```

## Project Structure

```
RLM/
├── main.py                 # CLI entry point
├── Dockerfile              # HTTP server container
├── docker-compose.yml      # Easy deployment
├── repl_server.py          # FastAPI + REPL server
└── rlm/
    ├── agent.py            # Main orchestration loop
    ├── prompts.py          # System prompt templates
    ├── parser.py           # Response parsing
    └── clients/
        ├── openrouter.py   # LLM API client
        ├── docker_sandbox.py # Local Docker client
        └── remote_sandbox.py # Remote HTTP client
```

## License

MIT
