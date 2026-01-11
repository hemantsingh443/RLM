"""
Remote Sandbox Client for RLM Engine.

HTTP client for communicating with a remote RLM sandbox server.
"""

import os
import tarfile
import tempfile
import requests
from typing import Optional, Dict, Any, List
from pathlib import Path


class RemoteSandbox:
    """
    Client for a remote RLM sandbox server.
    
    Communicates via HTTP API instead of Docker stdin/stdout.
    """
    
    def __init__(
        self,
        server_url: str,
        api_key: Optional[str] = None,
        timeout: int = 120
    ):
        """
        Initialize the remote sandbox client.
        
        Args:
            server_url: Base URL of the sandbox server (e.g., http://localhost:8080)
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key or os.environ.get("RLM_API_KEY", "")
        self.timeout = timeout
        
        self.headers = {}
        if self.api_key:
            self.headers["X-API-Key"] = self.api_key
    
    def _request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make an HTTP request to the server."""
        url = f"{self.server_url}{endpoint}"
        kwargs.setdefault('timeout', self.timeout)
        kwargs.setdefault('headers', {}).update(self.headers)
        
        response = requests.request(method, url, **kwargs)
        response.raise_for_status()
        return response
    
    def ping(self) -> bool:
        """Check if the server is reachable."""
        try:
            resp = self._request("GET", "/status")
            return resp.json().get("status") == "ready"
        except Exception:
            return False
    
    def get_status(self) -> Dict:
        """Get server status and file index info."""
        resp = self._request("GET", "/status")
        return resp.json()
    
    def exec_code(self, code: str) -> Dict[str, Any]:
        """
        Execute code on the remote sandbox.
        
        Args:
            code: Python code to execute
        
        Returns:
            Dict with 'success', 'output', and 'error' keys
        """
        try:
            resp = self._request("POST", "/execute", json={"code": code})
            return resp.json()
        except requests.RequestException as e:
            return {
                "success": False,
                "output": "",
                "error": f"Request failed: {e}"
            }
    
    def get_variable(self, name: str) -> Optional[Any]:
        """Get a variable's value from the sandbox."""
        try:
            resp = self._request("POST", "/get_var", json={"name": name})
            data = resp.json()
            if data.get("success"):
                return data.get("value")
            return None
        except Exception:
            return None
    
    def list_files(self, pattern: str = "*") -> List[str]:
        """List files in the sandbox."""
        try:
            resp = self._request("GET", f"/files?pattern={pattern}")
            return resp.json().get("files", [])
        except Exception:
            return []
    
    def read_file(self, path: str) -> Optional[str]:
        """Read a file from the sandbox."""
        try:
            resp = self._request("GET", f"/file/{path}")
            return resp.json().get("content")
        except Exception:
            return None
    
    def reindex(self) -> int:
        """Reindex the data directory."""
        try:
            resp = self._request("POST", "/reindex")
            return resp.json().get("files_indexed", 0)
        except Exception:
            return 0
    
    def reset(self) -> bool:
        """Reset the sandbox namespace."""
        try:
            resp = self._request("POST", "/reset")
            return resp.json().get("status") == "reset"
        except Exception:
            return False
    
    def upload_directory(self, local_path: str) -> bool:
        """
        Upload a local directory to the sandbox.
        
        Note: This requires the server to support file uploads,
        which may need additional implementation.
        For now, directories should be mounted via Docker volumes.
        
        Args:
            local_path: Path to local directory
        
        Returns:
            True if successful
        """
        # For remote deployment, directories are mounted via Docker volumes
        # This is a placeholder for future implementation
        raise NotImplementedError(
            "Direct upload not implemented. "
            "Mount the directory via Docker volumes or copy to server."
        )
    
    def start(self) -> bool:
        """
        Check connection (for interface compatibility with DockerSandbox).
        """
        return self.ping()
    
    def stop(self):
        """No-op for remote sandbox (server continues running)."""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
