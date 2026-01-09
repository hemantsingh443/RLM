"""
RLM Agent - Main Orchestration Logic.

Implements the recursive language model agent loop that coordinates
between the LLM (via OpenRouter) and the code execution sandbox (Docker).
"""

import os
from typing import List, Dict, Optional, Any

from .clients.openrouter import OpenRouterClient
from .clients.docker_sandbox import DockerSandbox
from .prompts import format_system_prompt
from .parser import (
    extract_first_code_block,
    detect_final_answer,
    truncate_output,
    format_execution_result
)


class RLMAgent:
    """
    Recursive Language Model Agent.
    
    Orchestrates the conversation between an LLM and a code execution
    sandbox to analyze large documents using recursive decomposition.
    """
    
    DEFAULT_MAX_TURNS = 15
    DEFAULT_TRUNCATION_LIMIT = 2000
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "xiaomi/mimo-v2-flash:free",
        max_turns: int = DEFAULT_MAX_TURNS,
        truncation_limit: int = DEFAULT_TRUNCATION_LIMIT,
        verbose: bool = True
    ):
        """
        Initialize the RLM Agent.
        
        Args:
            api_key: OpenRouter API key
            model: Model to use for root agent
            max_turns: Maximum conversation turns
            truncation_limit: Max chars for execution output
            verbose: Whether to print progress
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model
        self.max_turns = max_turns
        self.truncation_limit = truncation_limit
        self.verbose = verbose
        
        self.llm_client = OpenRouterClient(
            api_key=self.api_key,
            default_model=self.model
        )
        
        self.sandbox: Optional[DockerSandbox] = None
        self.history: List[Dict[str, str]] = []
        self.turn_count = 0
    
    def _log(self, message: str):
        """Print a log message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def _get_file_stats(self, file_path: str) -> Dict[str, Any]:
        """
        Get statistics about the context file.
        
        Args:
            file_path: Path to the context file
        
        Returns:
            Dict with file statistics
        """
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        return {
            'length': len(content),
            'words': len(content.split()),
            'lines': len(content.splitlines()),
            'path': file_path
        }
    
    def run(
        self,
        query: str,
        context_file: str,
        context_type: str = "text document"
    ) -> str:
        """
        Run the RLM agent on a query.
        
        Args:
            query: User's question/task
            context_file: Path to the context file
            context_type: Description of the context type
        
        Returns:
            Final answer string
        """
        self._log(f"\n{'='*60}")
        self._log(f"RLM Agent Starting")
        self._log(f"Query: {query}")
        self._log(f"Context: {context_file}")
        self._log(f"{'='*60}\n")
        
        # Get file statistics
        file_stats = self._get_file_stats(context_file)
        self._log(f"Context stats: {file_stats['length']} chars, {file_stats['words']} words, {file_stats['lines']} lines")
        
        # Initialize sandbox
        self._log("Starting Docker sandbox...")
        self.sandbox = DockerSandbox(
            context_file=context_file,
            api_key=self.api_key
        )
        
        # Build image if needed (first run)
        dockerfile_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not self._image_exists():
            self._log("Building Docker image (first run)...")
            if not self.sandbox.build_image(dockerfile_dir):
                return "Error: Failed to build Docker image"
        
        if not self.sandbox.start():
            return "Error: Failed to start sandbox"
        
        try:
            # Prepare system prompt
            system_prompt = format_system_prompt(
                context_length=file_stats['length'],
                context_type=context_type
            )
            
            # Initialize conversation history
            self.history = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}"}
            ]
            
            # Main agent loop
            for turn in range(self.max_turns):
                self.turn_count = turn + 1
                self._log(f"\n--- Turn {self.turn_count}/{self.max_turns} ---")
                
                # Call root LLM
                response = self.llm_client.chat(self.history)
                self._log(f"Model response:\n{response[:500]}{'...' if len(response) > 500 else ''}")
                
                # Extract code block first (we may need to execute it before FINAL_VAR)
                code_block = extract_first_code_block(response)
                
                # Check for final answer
                is_final, answer_type, answer_content = detect_final_answer(response)
                
                if code_block:
                    self._log(f"Executing code:\n{code_block[:300]}{'...' if len(code_block) > 300 else ''}")
                    
                    # Execute in sandbox
                    result = self.sandbox.exec_code(code_block)
                    
                    # Format and truncate output
                    formatted_result = format_execution_result(result)
                    truncated_result = truncate_output(formatted_result, self.truncation_limit)
                    
                    self._log(f"Execution result ({len(formatted_result)} chars):\n{truncated_result[:500]}")
                    
                    # Now check if there was a FINAL answer after the code
                    if is_final:
                        self._log(f"\nFinal answer detected ({answer_type})")
                        
                        if answer_type == 'FINAL_VAR':
                            # Retrieve variable value from sandbox (code was just executed)
                            value = self.sandbox.get_variable(answer_content)
                            if value is not None:
                                return str(value)
                            else:
                                # Variable not found, but we have the execution result
                                return f"Variable '{answer_content}' not found. Last execution output:\n{truncated_result}"
                        else:
                            return answer_content
                    
                    # Update conversation history
                    self.history.append({"role": "assistant", "content": response})
                    self.history.append({
                        "role": "user",
                        "content": f"Execution Result:\n{truncated_result}"
                    })
                
                elif is_final:
                    # No code block, but we have a final answer (pure FINAL)
                    self._log(f"\nFinal answer detected ({answer_type})")
                    
                    if answer_type == 'FINAL_VAR':
                        # Retrieve variable value from sandbox
                        value = self.sandbox.get_variable(answer_content)
                        if value is not None:
                            return str(value)
                        else:
                            return f"Error: Variable '{answer_content}' not found"
                    else:
                        return answer_content
                
                else:
                    # Pure reasoning step - no code and no final answer
                    self._log("No code block found, treating as reasoning step")
                    
                    self.history.append({"role": "assistant", "content": response})
                    self.history.append({
                        "role": "user",
                        "content": "Continue with your analysis. Execute code or provide the final answer using FINAL() or FINAL_VAR()."
                    })
            
            # Max turns reached
            self._log("\nMax turns reached without final answer")
            return "Error: Maximum turns reached without final answer. Last response:\n" + response
        
        finally:
            # Cleanup
            self._log("\nStopping sandbox...")
            if self.sandbox:
                self.sandbox.stop()
    
    def _image_exists(self) -> bool:
        """Check if the Docker image already exists."""
        import subprocess
        try:
            result = subprocess.run(
                ["docker", "images", "-q", DockerSandbox.IMAGE_NAME],
                capture_output=True,
                text=True
            )
            return bool(result.stdout.strip())
        except Exception:
            return False
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.history.copy()
    
    def get_turn_count(self) -> int:
        """Get the number of turns completed."""
        return self.turn_count
