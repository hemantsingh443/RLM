"""
Docker Sandbox Client for RLM Engine.

Manages the Docker container lifecycle and provides an interface
for executing code in the persistent REPL.
"""

import subprocess
import json
import os
import threading
import queue
from typing import Optional, Dict, Any


class DockerSandbox:
    """
    Manages a Docker container running the REPL server.
    
    The container maintains a persistent Python namespace where the
    LLM can execute code and build up state across multiple turns.
    """
    
    IMAGE_NAME = "rlm-sandbox"
    CONTAINER_NAME = "rlm-sandbox-instance"
    
    def __init__(
        self,
        context_file: Optional[str] = None,
        api_key: Optional[str] = None,
        max_recursion_depth: int = 3,
        verbose: bool = True
    ):
        """
        Initialize the Docker sandbox manager.
        
        Args:
            context_file: Path to the context file to mount
            api_key: OpenRouter API key for sub-calls
            max_recursion_depth: Max depth for llm_query recursion
            verbose: Print stderr logs from container
        """
        self.context_file = context_file
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.max_recursion_depth = max_recursion_depth
        self.verbose = verbose
        
        self.process: Optional[subprocess.Popen] = None
        self.stdout_queue: Optional[queue.Queue] = None
        self._stdout_reader: Optional[threading.Thread] = None
        self._stderr_reader: Optional[threading.Thread] = None
        self._running = False
    
    def build_image(self, dockerfile_dir: str = ".") -> bool:
        """
        Build the Docker image from the Dockerfile.
        
        Args:
            dockerfile_dir: Directory containing the Dockerfile
        
        Returns:
            True if build succeeded
        """
        try:
            result = subprocess.run(
                ["docker", "build", "-t", self.IMAGE_NAME, dockerfile_dir],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                print(f"Docker build failed: {result.stderr}")
                return False
            return True
        except subprocess.TimeoutExpired:
            print("Docker build timed out")
            return False
        except FileNotFoundError:
            print("Docker not found. Please install Docker.")
            return False
    
    def start(self) -> bool:
        """
        Start the Docker container with the REPL server.
        
        Returns:
            True if container started successfully
        """
        # Stop any existing container
        self.stop()
        
        # Build the docker run command
        cmd = [
            "docker", "run",
            "-i",  # Interactive (keep stdin open)
            "--rm",  # Remove container when stopped
            "--name", self.CONTAINER_NAME,
        ]
        
        # Mount context file if provided
        if self.context_file and os.path.exists(self.context_file):
            abs_path = os.path.abspath(self.context_file)
            cmd.extend(["-v", f"{abs_path}:/mnt/data/input.txt:ro"])
        
        # Pass environment variables
        if self.api_key:
            cmd.extend(["-e", f"OPENROUTER_API_KEY={self.api_key}"])
        
        cmd.extend(["-e", f"RLM_MAX_RECURSION_DEPTH={self.max_recursion_depth}"])
        cmd.extend(["-e", "RLM_RECURSION_DEPTH=0"])
        
        # Add the image name
        cmd.append(self.IMAGE_NAME)
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # Separate stderr
                text=True,
                bufsize=1
            )
            
            # Set up output queue and reader threads
            self.stdout_queue = queue.Queue()
            self._running = True
            
            # Start stdout reader (JSON responses)
            self._stdout_reader = threading.Thread(
                target=self._read_stream,
                args=(self.process.stdout, self.stdout_queue, "stdout"),
                daemon=True
            )
            self._stdout_reader.start()
            
            # Start stderr reader (logs only - print if verbose)
            self._stderr_reader = threading.Thread(
                target=self._read_stderr,
                daemon=True
            )
            self._stderr_reader.start()
            
            # Wait for ready signal
            response = self._read_response(timeout=30)
            if response and response.get("status") == "ready":
                context_info = response.get("context_info", "")
                print(f"Sandbox started: {response.get('message')} ({context_info})")
                return True
            else:
                print(f"Unexpected startup response: {response}")
                return False
            
        except Exception as e:
            print(f"Failed to start container: {e}")
            return False
    
    def _read_stream(self, stream, output_queue, name: str):
        """Background thread to read from a stream."""
        while self._running and stream:
            try:
                line = stream.readline()
                if line:
                    output_queue.put(line.strip())
                elif self.process and self.process.poll() is not None:
                    break
            except Exception:
                break
    
    def _read_stderr(self):
        """Background thread to read and optionally print stderr."""
        while self._running and self.process and self.process.stderr:
            try:
                line = self.process.stderr.readline()
                if line:
                    if self.verbose:
                        print(f"[Sandbox]: {line.strip()}")
                elif self.process.poll() is not None:
                    break
            except Exception:
                break
    
    def _read_response(self, timeout: float = 60) -> Optional[Dict]:
        """
        Read a JSON response from stdout.
        
        Args:
            timeout: Maximum time to wait for response
        
        Returns:
            Parsed JSON response or None
        """
        try:
            line = self.stdout_queue.get(timeout=timeout)
            return json.loads(line)
        except queue.Empty:
            return None
        except json.JSONDecodeError as e:
            print(f"Invalid JSON from container: {e} - received: {line[:100]}")
            return None
    
    def exec_code(self, code: str, timeout: float = 120) -> Dict[str, Any]:
        """
        Execute code in the sandbox and return the result.
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
        
        Returns:
            Dict with 'success', 'output', and 'error' keys
        """
        if not self.process or self.process.poll() is not None:
            return {
                "success": False,
                "output": "",
                "error": "Container is not running"
            }
        
        command = json.dumps({"action": "execute", "code": code})
        
        try:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
            
            response = self._read_response(timeout=timeout)
            if response is None:
                return {
                    "success": False,
                    "output": "",
                    "error": "Timeout waiting for execution result"
                }
            
            return response
            
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"Execution error: {str(e)}"
            }
    
    def get_variable(self, name: str) -> Optional[Any]:
        """
        Get the value of a variable from the sandbox namespace.
        
        Args:
            name: Variable name
        
        Returns:
            Variable value or None
        """
        if not self.process or self.process.poll() is not None:
            return None
        
        command = json.dumps({"action": "get_var", "name": name})
        
        try:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
            
            response = self._read_response(timeout=10)
            if response and response.get("success"):
                return response.get("value")
            return None
            
        except Exception:
            return None
    
    def ping(self) -> bool:
        """Check if the container is responsive."""
        if not self.process or self.process.poll() is not None:
            return False
        
        command = json.dumps({"action": "ping"})
        
        try:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
            
            response = self._read_response(timeout=5)
            return response is not None and response.get("success", False)
            
        except Exception:
            return False
    
    def stop(self):
        """Stop the Docker container."""
        self._running = False
        
        if self.process:
            try:
                # Send shutdown command
                shutdown_cmd = json.dumps({"action": "shutdown"})
                self.process.stdin.write(shutdown_cmd + "\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except Exception:
                pass
            
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            
            self.process = None
        
        # Force remove container if it exists
        try:
            subprocess.run(
                ["docker", "rm", "-f", self.CONTAINER_NAME],
                capture_output=True,
                timeout=10
            )
        except Exception:
            pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup."""
        self.stop()
        return False
