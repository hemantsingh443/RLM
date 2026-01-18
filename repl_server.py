#!/usr/bin/env python3
"""
RLM Sandbox Server - Persistent Python REPL with HTTP API.

This server provides:
1. HTTP API for remote code execution
2. Directory indexing and file access
3. Persistent global namespace across executions
4. llm_query() for recursive sub-agent calls
"""

import sys
import os
import io
import json
import traceback
import glob
import fnmatch
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import redirect_stdout, redirect_stderr

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ============================================================================
# Configuration
# ============================================================================

MAX_OUTPUT_SIZE = 50000  # characters
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB per file
DATA_DIR = "/mnt/data"
API_KEY = os.environ.get("RLM_API_KEY", "")  # Optional auth

# ============================================================================
# Global State
# ============================================================================

# Persistent namespace for code execution
global_namespace: Dict[str, Any] = {}

# File index: path -> metadata
file_index: Dict[str, Dict] = {}

# ============================================================================
# Helper Functions (exposed to LLM-generated code)
# ============================================================================

def log(message: str):
    """Log to stderr."""
    sys.stderr.write(f"[RLM] {message}\n")
    sys.stderr.flush()


def list_files(pattern: str = "*") -> List[str]:
    """
    List all indexed files, optionally filtered by glob pattern.
    
    Args:
        pattern: Glob pattern (e.g., "*.py", "src/**/*.js")
    
    Returns:
        List of file paths
    """
    all_files = list(file_index.keys())
    if pattern == "*":
        return sorted(all_files)
    return sorted([f for f in all_files if fnmatch.fnmatch(f, pattern)])


def get_file_tree() -> Dict:
    """
    Get the file tree with metadata.
    
    Returns:
        Nested dict representing directory structure
    """
    tree = {}
    for path, meta in file_index.items():
        parts = path.split('/')
        current = tree
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = meta
    return tree


def read_file(path: str) -> str:
    """
    Read a file's content.
    
    Args:
        path: Relative path from the data directory
    
    Returns:
        File content as string
    """
    if path not in file_index:
        # Try to find it
        full_path = os.path.join(DATA_DIR, path)
        if os.path.exists(full_path):
            try:
                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    return f.read()
            except Exception as e:
                return f"Error reading file: {e}"
        return f"File not found: {path}"
    
    full_path = os.path.join(DATA_DIR, path)
    try:
        with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


def search_files(pattern: str, file_pattern: str = "*") -> List[Dict]:
    """
    Search for a regex pattern across all files.
    
    Args:
        pattern: Regex pattern to search for
        file_pattern: Glob pattern to filter files (e.g., "*.py")
    
    Returns:
        List of matches with file, line number, and content
    """
    matches = []
    regex = re.compile(pattern)
    
    for path in list_files(file_pattern):
        content = read_file(path)
        for i, line in enumerate(content.split('\n'), 1):
            if regex.search(line):
                matches.append({
                    'file': path,
                    'line': i,
                    'content': line.strip()[:200]
                })
                if len(matches) >= 100:  # Limit results
                    return matches
    return matches


def llm_query(prompt: str, model: str = "xiaomi/mimo-v2-flash:free") -> str:
    """
    Make a recursive LLM sub-call with context about available files.
    
    This is a simplified version that makes a single LLM call with file context,
    rather than spawning a full agent loop (which was causing timeouts).
    
    Args:
        prompt: The task/question for the sub-agent
        model: Model to use
    
    Returns:
        LLM response
    """
    import requests
    
    log(f"llm_query called with prompt: {prompt[:100]}...")
    
    # Check recursion depth
    current_depth = int(os.environ.get("RLM_RECURSION_DEPTH", "0"))
    max_depth = int(os.environ.get("RLM_MAX_RECURSION_DEPTH", "3"))
    
    if current_depth >= max_depth:
        log(f"Max recursion depth reached: {current_depth}")
        return f"Error: Max recursion depth ({max_depth}) reached."
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        log("No API key found")
        return "Error: OPENROUTER_API_KEY not set"
    
    # Increment recursion depth
    os.environ["RLM_RECURSION_DEPTH"] = str(current_depth + 1)
    log(f"Recursion depth now: {current_depth + 1}")
    
    try:
        # Build context about available files
        file_list = list(file_index.keys())[:20]  # Limit to first 20
        file_context = f"Available files: {file_list}"
        
        # Enhanced prompt with file context
        enhanced_prompt = f"""You are a helpful assistant analyzing code/documents.

{file_context}

User request: {prompt}

Provide a direct, helpful answer based on the information given."""

        log(f"Making LLM request...")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": enhanced_prompt}],
            "max_tokens": 1000
        }
        
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=60  # 60 second timeout for sub-calls
        )
        resp.raise_for_status()
        
        result = resp.json()['choices'][0]['message']['content']
        log(f"LLM response received: {len(result)} chars")
        return result
        
    except requests.Timeout:
        log("LLM request timed out")
        return "Error: LLM request timed out"
    except Exception as e:
        log(f"LLM request error: {e}")
        return f"Error: {e}"
    finally:
        # Restore recursion depth
        os.environ["RLM_RECURSION_DEPTH"] = str(current_depth)


# ============================================================================
# Code Execution
# ============================================================================

def execute_code(code: str) -> Dict:
    """Execute code in the persistent namespace."""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, global_namespace)
        
        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()
        
        if len(stdout) > MAX_OUTPUT_SIZE:
            stdout = stdout[:MAX_OUTPUT_SIZE] + f"\n... [Truncated at {MAX_OUTPUT_SIZE} chars]"
        
        return {
            "success": True,
            "output": stdout,
            "error": stderr if stderr else None
        }
    except Exception as e:
        return {
            "success": False,
            "output": stdout_capture.getvalue(),
            "error": traceback.format_exc()
        }


def index_directory(path: str = DATA_DIR) -> Dict[str, Dict]:
    """Index all files in the data directory."""
    global file_index
    file_index = {}
    
    if not os.path.exists(path):
        log(f"Data directory not found: {path}")
        return file_index
    
    for root, dirs, files in os.walk(path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for fname in files:
            if fname.startswith('.'):
                continue
            
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, path)
            
            try:
                stat = os.stat(full_path)
                size = stat.st_size
                
                # Skip large files
                if size > MAX_FILE_SIZE:
                    continue
                
                # Detect file type
                ext = os.path.splitext(fname)[1].lower()
                file_type = {
                    '.py': 'python',
                    '.js': 'javascript',
                    '.ts': 'typescript',
                    '.jsx': 'react',
                    '.tsx': 'react',
                    '.java': 'java',
                    '.c': 'c',
                    '.cpp': 'cpp',
                    '.h': 'header',
                    '.go': 'go',
                    '.rs': 'rust',
                    '.rb': 'ruby',
                    '.php': 'php',
                    '.md': 'markdown',
                    '.txt': 'text',
                    '.json': 'json',
                    '.yaml': 'yaml',
                    '.yml': 'yaml',
                    '.xml': 'xml',
                    '.html': 'html',
                    '.css': 'css',
                    '.sql': 'sql',
                    '.sh': 'shell',
                }.get(ext, 'text')
                
                file_index[rel_path] = {
                    'size': size,
                    'type': file_type,
                    'ext': ext
                }
            except Exception as e:
                log(f"Error indexing {full_path}: {e}")
    
    log(f"Indexed {len(file_index)} files")
    return file_index


def initialize_namespace():
    """Initialize the global namespace with helper functions."""
    global_namespace['list_files'] = list_files
    global_namespace['get_file_tree'] = get_file_tree
    global_namespace['read_file'] = read_file
    global_namespace['search_files'] = search_files
    global_namespace['llm_query'] = llm_query
    global_namespace['files'] = file_index


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(title="RLM Sandbox Server", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class ExecuteRequest(BaseModel):
    code: str
    
class ExecuteResponse(BaseModel):
    success: bool
    output: str
    error: Optional[str] = None

class StatusResponse(BaseModel):
    status: str
    files_indexed: int
    namespace_vars: List[str]

class GetVarRequest(BaseModel):
    name: str


# Auth dependency
async def verify_api_key(x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


@app.get("/status", response_model=StatusResponse)
async def get_status(auth: bool = Depends(verify_api_key)):
    """Health check and status."""
    user_vars = [k for k in global_namespace.keys() 
                 if not k.startswith('_') and k not in 
                 ('list_files', 'get_file_tree', 'read_file', 'search_files', 'llm_query', 'files')]
    return StatusResponse(
        status="ready",
        files_indexed=len(file_index),
        namespace_vars=user_vars
    )


@app.post("/execute", response_model=ExecuteResponse)
async def execute(request: ExecuteRequest, auth: bool = Depends(verify_api_key)):
    """Execute Python code in the persistent REPL."""
    log(f"Executing code ({len(request.code)} chars)")
    result = execute_code(request.code)
    return ExecuteResponse(**result)


@app.get("/files")
async def get_files(pattern: str = "*", auth: bool = Depends(verify_api_key)):
    """List indexed files."""
    return {"files": list_files(pattern)}


@app.get("/file/{path:path}")
async def get_file(path: str, auth: bool = Depends(verify_api_key)):
    """Read a specific file."""
    content = read_file(path)
    if content.startswith("Error") or content.startswith("File not found"):
        raise HTTPException(status_code=404, detail=content)
    return {"path": path, "content": content}


@app.post("/reindex")
async def reindex(auth: bool = Depends(verify_api_key)):
    """Reindex the data directory."""
    index_directory()
    global_namespace['files'] = file_index
    return {"files_indexed": len(file_index)}


@app.post("/get_var")
async def get_variable(request: GetVarRequest, auth: bool = Depends(verify_api_key)):
    """Get a variable from the namespace."""
    if request.name not in global_namespace:
        raise HTTPException(status_code=404, detail=f"Variable '{request.name}' not found")
    
    value = global_namespace[request.name]
    try:
        json.dumps(value)
        return {"success": True, "value": value}
    except (TypeError, ValueError):
        return {"success": True, "value": repr(value)}


@app.post("/reset")
async def reset_namespace(auth: bool = Depends(verify_api_key)):
    """Reset the namespace (clear user variables)."""
    global global_namespace
    global_namespace = {}
    initialize_namespace()
    return {"status": "reset"}


# ============================================================================
# Startup
# ============================================================================

@app.on_event("startup")
async def startup():
    log("Starting RLM Sandbox Server...")
    index_directory()
    initialize_namespace()
    log(f"Ready. Indexed {len(file_index)} files.")


def main():
    """Run the server."""
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    
    log(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
