#!/usr/bin/env python3
"""
Persistent Python REPL Server for RLM Sandbox.

This server maintains a global namespace across multiple code executions,
allowing the LLM to build up state over multiple turns. It communicates
via JSON over stdout, with logs going to stderr.
"""

import sys
import os
import io
import json
import traceback
from contextlib import redirect_stdout, redirect_stderr

# Maximum output size to prevent memory explosion
MAX_OUTPUT_SIZE = 50000  # characters

# Global namespace for persistent state
global_namespace = {}

# Use stderr for all logging to keep stdout clean for JSON only
def log(message):
    """Log a message to stderr (won't interfere with JSON on stdout)."""
    sys.stderr.write(f"[REPL] {message}\n")
    sys.stderr.flush()


def llm_query(prompt: str, model: str = "xiaomi/mimo-v2-flash:free") -> str:
    """
    Synchronous call to OpenRouter API for recursive sub-agent calls.
    
    Args:
        prompt: The prompt to send to the LLM
        model: The model to use (defaults to free model)
    
    Returns:
        The LLM response text
    """
    import requests
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return "Error: OPENROUTER_API_KEY not set in environment"
    
    # Recursion depth guard
    current_depth = int(os.environ.get("RLM_RECURSION_DEPTH", "0"))
    max_depth = int(os.environ.get("RLM_MAX_RECURSION_DEPTH", "3"))
    
    if current_depth >= max_depth:
        return f"Error: Maximum recursion depth ({max_depth}) reached. Cannot make more llm_query calls."
    
    # Increment recursion depth for nested calls
    os.environ["RLM_RECURSION_DEPTH"] = str(current_depth + 1)
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/rlm-engine",
            "X-Title": "RLM Engine Sub-Query"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        log(f"Making LLM query with model {model}...")
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=120
        )
        resp.raise_for_status()
        
        result = resp.json()
        return result['choices'][0]['message']['content']
    
    except requests.exceptions.RequestException as e:
        return f"Error making LLM request: {str(e)}"
    except (KeyError, IndexError) as e:
        return f"Error parsing LLM response: {str(e)}"
    finally:
        # Decrement recursion depth after call completes
        os.environ["RLM_RECURSION_DEPTH"] = str(current_depth)


def load_context():
    """
    Load the context file from the mounted volume.
    Returns info message to include in ready response.
    """
    context_path = "/mnt/data/input.txt"
    
    if os.path.exists(context_path):
        try:
            with open(context_path, 'r', encoding='utf-8', errors='replace') as f:
                context = f.read()
            global_namespace['context'] = context
            msg = f"Loaded context with {len(context)} characters ({len(context.split())} words)"
            log(msg)
            return msg
        except Exception as e:
            msg = f"Error loading context: {e}"
            log(msg)
            return msg
    else:
        log("No context file found at /mnt/data/input.txt")
        global_namespace['context'] = ""
        return "No context file found"


def execute_code(code: str) -> dict:
    """
    Execute code in the persistent global namespace.
    
    Args:
        code: Python code to execute
    
    Returns:
        dict with 'output', 'error', and 'success' keys
    """
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Execute the code in our persistent namespace
            exec(code, global_namespace)
        
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()
        
        # Truncate if needed
        if len(stdout_output) > MAX_OUTPUT_SIZE:
            stdout_output = stdout_output[:MAX_OUTPUT_SIZE] + f"\n... [Output truncated at {MAX_OUTPUT_SIZE} chars]"
        
        return {
            "success": True,
            "output": stdout_output,
            "error": stderr_output if stderr_output else None
        }
    
    except Exception as e:
        error_trace = traceback.format_exc()
        return {
            "success": False,
            "output": stdout_capture.getvalue(),
            "error": error_trace
        }


def send_response(data: dict):
    """Send a JSON response to stdout."""
    sys.stdout.write(json.dumps(data) + "\n")
    sys.stdout.flush()


def main():
    """
    Main REPL loop. Reads JSON commands from stdin, executes them,
    and writes JSON responses to stdout.
    """
    # Initialize the namespace with helper functions
    # Use the built-in print - it will be captured by redirect_stdout during exec
    global_namespace['llm_query'] = llm_query
    
    # Load context if available
    context_msg = load_context()
    
    # Signal ready with JSON on stdout
    send_response({
        "status": "ready", 
        "message": "RLM Sandbox initialized",
        "context_info": context_msg
    })
    
    # Main loop
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            command = json.loads(line)
        except json.JSONDecodeError as e:
            send_response({
                "success": False,
                "error": f"Invalid JSON: {e}"
            })
            continue
        
        action = command.get("action", "execute")
        
        if action == "execute":
            code = command.get("code", "")
            log(f"Executing code ({len(code)} chars)")
            result = execute_code(code)
            send_response(result)
        
        elif action == "get_var":
            var_name = command.get("name", "")
            if var_name in global_namespace:
                value = global_namespace[var_name]
                # Try to serialize, fall back to repr
                try:
                    json.dumps(value)
                    result = {"success": True, "value": value}
                except (TypeError, ValueError):
                    result = {"success": True, "value": repr(value)}
            else:
                result = {"success": False, "error": f"Variable '{var_name}' not found"}
            send_response(result)
        
        elif action == "list_vars":
            # List all user-defined variables (exclude builtins and internals)
            user_vars = {
                k: type(v).__name__ 
                for k, v in global_namespace.items() 
                if not k.startswith('_') and k not in ('llm_query', 'context')
            }
            result = {"success": True, "variables": user_vars}
            send_response(result)
        
        elif action == "ping":
            result = {"success": True, "message": "pong"}
            send_response(result)
        
        elif action == "shutdown":
            result = {"success": True, "message": "Shutting down"}
            send_response(result)
            break
        
        else:
            result = {"success": False, "error": f"Unknown action: {action}"}
            send_response(result)


if __name__ == "__main__":
    main()
