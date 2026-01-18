"""
Self Sandbox Client for RLM Engine.

In-process sandbox for sub-agents running inside the container.
Executes code directly in the shared global namespace without network calls.
"""

import io
import json
import traceback
from typing import Dict, Any, Optional, Callable
from contextlib import redirect_stdout, redirect_stderr


class SelfSandbox:
    """
    In-process sandbox for recursive sub-agents.
    
    When a sub-agent is spawned inside the container via llm_query(),
    it uses this sandbox to execute code directly in the shared
    global namespace. This avoids HTTP overhead and allows sub-agents
    to share state with the parent execution context.
    """
    
    MAX_OUTPUT_SIZE = 50000  # characters
    
    def __init__(
        self,
        namespace: Dict[str, Any],
        file_index: Dict[str, Dict],
        helper_functions: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize the in-process sandbox.
        
        Args:
            namespace: The shared global namespace for code execution
            file_index: The file index dictionary
            helper_functions: Optional dict of helper functions to inject
                             (list_files, read_file, search_files, etc.)
        """
        self.namespace = namespace
        self.file_index = file_index
        self.helper_functions = helper_functions or {}
        
        # Ensure helper functions are in namespace
        for name, func in self.helper_functions.items():
            if name not in self.namespace:
                self.namespace[name] = func
    
    def exec_code(self, code: str, timeout: float = 120) -> Dict[str, Any]:
        """
        Execute code directly in the shared namespace.
        
        Args:
            code: Python code to execute
            timeout: Ignored (no timeout for in-process execution)
        
        Returns:
            Dict with 'success', 'output', and 'error' keys
        """
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, self.namespace)
            
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()
            
            # Truncate if too long
            if len(stdout) > self.MAX_OUTPUT_SIZE:
                stdout = stdout[:self.MAX_OUTPUT_SIZE] + \
                         f"\n... [Truncated at {self.MAX_OUTPUT_SIZE} chars]"
            
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
    
    def get_variable(self, name: str) -> Optional[Any]:
        """
        Get a variable's value from the namespace.
        
        Args:
            name: Variable name
        
        Returns:
            Variable value or None if not found
        """
        if name not in self.namespace:
            return None
        
        value = self.namespace[name]
        
        # Try to make it JSON-serializable
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return repr(value)
    
    def ping(self) -> bool:
        """Always returns True - in-process sandbox is always ready."""
        return True
    
    def start(self) -> bool:
        """
        Start the sandbox (no-op for in-process).
        
        Returns:
            Always True
        """
        return True
    
    def stop(self):
        """Stop the sandbox (no-op for in-process)."""
        pass
    
    def reindex(self) -> int:
        """
        Reindex is not supported for in-process sandbox.
        The file index is managed by the parent server.
        
        Returns:
            Current file count
        """
        return len(self.file_index)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
