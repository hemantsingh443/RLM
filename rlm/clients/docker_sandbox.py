"""
Docker Sandbox Client for RLM Engine.

Manages the Docker container lifecycle and provides an interface
for executing code via HTTP API (delegating to RemoteSandbox).
"""

import subprocess
import time
import os
from typing import Optional, Dict, Any

from .remote_sandbox import RemoteSandbox


class DockerSandbox:
    """
    Manages a Docker container running the RLM sandbox server.
    
    Starts the container with HTTP port exposed and delegates all
    operations to RemoteSandbox for actual API communication.
    """
    
    IMAGE_NAME = "rlm-sandbox"
    CONTAINER_NAME = "rlm-sandbox-instance"
    DEFAULT_PORT = 8080
    
    def __init__(
        self,
        context_file: Optional[str] = None,
        api_key: Optional[str] = None,
        max_recursion_depth: int = 3,
        verbose: bool = True,
        port: int = DEFAULT_PORT
    ):
        """
        Initialize the Docker sandbox manager.
        
        Args:
            context_file: Path to context file or directory to mount
            api_key: OpenRouter API key for recursive sub-calls
            max_recursion_depth: Max depth for llm_query recursion
            verbose: Print progress logs
            port: Host port to expose (default: 8080)
        """
        self.context_file = context_file
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.max_recursion_depth = max_recursion_depth
        self.verbose = verbose
        self.port = port
        
        # RemoteSandbox handles actual API communication
        self._remote: Optional[RemoteSandbox] = None
        self._container_id: Optional[str] = None
    
    def _log(self, message: str):
        """Print log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[DockerSandbox] {message}")
    
    def build_image(self, dockerfile_dir: str = ".") -> bool:
        """
        Build the Docker image from the Dockerfile.
        
        Args:
            dockerfile_dir: Directory containing the Dockerfile
        
        Returns:
            True if build succeeded
        """
        self._log(f"Building image from {dockerfile_dir}...")
        try:
            result = subprocess.run(
                ["docker", "build", "-t", self.IMAGE_NAME, dockerfile_dir],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                self._log(f"Build failed: {result.stderr}")
                return False
            self._log("Image built successfully")
            return True
        except subprocess.TimeoutExpired:
            self._log("Build timed out")
            return False
        except FileNotFoundError:
            self._log("Docker not found. Please install Docker.")
            return False
    
    def start(self) -> bool:
        """
        Start the Docker container with HTTP server exposed.
        
        Returns:
            True if container started and is responding
        """
        # Stop any existing container first
        self.stop()
        
        # Build docker run command
        cmd = [
            "docker", "run",
            "-d",  # Detached mode
            "--rm",  # Remove when stopped
            "--name", self.CONTAINER_NAME,
            "-p", f"{self.port}:8080",  # Expose HTTP port
        ]
        
        # Mount context file or directory
        if self.context_file and os.path.exists(self.context_file):
            abs_path = os.path.abspath(self.context_file)
            if os.path.isdir(abs_path):
                # Mount directory
                cmd.extend(["-v", f"{abs_path}:/mnt/data:ro"])
                self._log(f"Mounting directory: {abs_path}")
            else:
                # Mount single file
                cmd.extend(["-v", f"{abs_path}:/mnt/data/input.txt:ro"])
                self._log(f"Mounting file: {abs_path}")
        
        # Pass environment variables
        if self.api_key:
            cmd.extend(["-e", f"OPENROUTER_API_KEY={self.api_key}"])
        cmd.extend(["-e", f"RLM_MAX_RECURSION_DEPTH={self.max_recursion_depth}"])
        cmd.extend(["-e", "RLM_RECURSION_DEPTH=0"])
        
        # Add image name
        cmd.append(self.IMAGE_NAME)
        
        try:
            self._log("Starting container...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                self._log(f"Failed to start container: {result.stderr}")
                return False
            
            self._container_id = result.stdout.strip()[:12]
            self._log(f"Container started: {self._container_id}")
            
            # Wait for HTTP server to be ready
            return self._wait_for_ready()
            
        except subprocess.TimeoutExpired:
            self._log("Container start timed out")
            return False
        except Exception as e:
            self._log(f"Error starting container: {e}")
            return False
    
    def _wait_for_ready(self, timeout: float = 30, poll_interval: float = 0.5) -> bool:
        """
        Wait for the HTTP server to become ready.
        
        Args:
            timeout: Maximum time to wait in seconds
            poll_interval: Time between health checks
        
        Returns:
            True if server is ready
        """
        self._log(f"Waiting for server on port {self.port}...")
        
        # Create RemoteSandbox client - long timeout for sub-agent LLM calls
        self._remote = RemoteSandbox(
            server_url=f"http://localhost:{self.port}",
            timeout=300  # Sub-agents may need multiple LLM calls
        )
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._remote.ping():
                self._log("Server is ready")
                return True
            time.sleep(poll_interval)
        
        self._log("Server failed to become ready")
        return False
    
    def exec_code(self, code: str, timeout: float = 120) -> Dict[str, Any]:
        """
        Execute code in the sandbox.
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time
        
        Returns:
            Dict with 'success', 'output', and 'error' keys
        """
        if not self._remote:
            return {
                "success": False,
                "output": "",
                "error": "Container is not running"
            }
        return self._remote.exec_code(code)
    
    def get_variable(self, name: str) -> Optional[Any]:
        """
        Get a variable's value from the sandbox namespace.
        
        Args:
            name: Variable name
        
        Returns:
            Variable value or None
        """
        if not self._remote:
            return None
        return self._remote.get_variable(name)
    
    def ping(self) -> bool:
        """Check if the container is responsive."""
        if not self._remote:
            return False
        return self._remote.ping()
    
    def reindex(self) -> int:
        """Reindex the data directory."""
        if not self._remote:
            return 0
        return self._remote.reindex()
    
    def stop(self):
        """Stop and remove the Docker container."""
        self._remote = None
        
        # Force remove container if it exists
        try:
            subprocess.run(
                ["docker", "rm", "-f", self.CONTAINER_NAME],
                capture_output=True,
                timeout=10
            )
        except Exception:
            pass
        
        self._container_id = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup."""
        self.stop()
        return False
