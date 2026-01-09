"""
Response Parser for RLM Engine.

Handles parsing of LLM responses to extract code blocks,
detect final answers, and manage output truncation.
"""

import re
from typing import Optional, Tuple, List


def extract_code_blocks(response: str) -> List[str]:
    """
    Extract Python code blocks from a response.
    
    Args:
        response: LLM response text
    
    Returns:
        List of code block contents
    """
    # Match ```python ... ``` blocks
    pattern = r'```python\s*\n(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    
    # Also try generic code blocks if no python-specific ones found
    if not matches:
        pattern = r'```\s*\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
    
    return [block.strip() for block in matches]


def extract_first_code_block(response: str) -> Optional[str]:
    """
    Extract the first Python code block from a response.
    
    Args:
        response: LLM response text
    
    Returns:
        Code block content or None
    """
    blocks = extract_code_blocks(response)
    return blocks[0] if blocks else None


def detect_final_answer(response: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Detect if the response contains a final answer.
    
    The format must be:
    - FINAL(answer text here) - at start of line or after whitespace
    - FINAL_VAR(variable_name) - at start of line or after whitespace
    
    This avoids false positives like "FINAL ANALYSIS:" or "FINALLY:"
    
    Args:
        response: LLM response text
    
    Returns:
        Tuple of (is_final, answer_type, answer_content)
        - is_final: True if a final answer was detected
        - answer_type: 'FINAL' or 'FINAL_VAR' or None
        - answer_content: The answer content or variable name
    """
    # Check for FINAL_VAR(...) pattern first (more specific)
    # Must be at start of line or after whitespace, immediately followed by (
    final_var_match = re.search(r'(?:^|\s)FINAL_VAR\((\w+)\)', response, re.MULTILINE)
    if final_var_match:
        return True, 'FINAL_VAR', final_var_match.group(1)
    
    # Check for FINAL(...) pattern
    # Must be at start of line or after whitespace, immediately followed by (
    # The opening paren must immediately follow FINAL (no space)
    final_match = re.search(r'(?:^|\s)FINAL\(([^)]+)\)', response, re.MULTILINE | re.DOTALL)
    if final_match:
        content = final_match.group(1).strip()
        # Make sure it's not empty
        if content:
            return True, 'FINAL', content
    
    return False, None, None


def truncate_output(output: str, max_chars: int = 2000) -> str:
    """
    Truncate output to fit within context limits.
    
    Args:
        output: Raw output string
        max_chars: Maximum characters to keep
    
    Returns:
        Truncated output with indicator
    """
    if len(output) <= max_chars:
        return output
    
    # Keep first portion and indicate truncation
    truncated = output[:max_chars]
    
    # Try to truncate at a natural boundary (newline)
    last_newline = truncated.rfind('\n')
    if last_newline > max_chars * 0.7:  # If there's a newline in the last 30%
        truncated = truncated[:last_newline]
    
    return truncated + f"\n\n... [Output truncated. Total length: {len(output)} chars]"


def format_execution_result(result: dict) -> str:
    """
    Format an execution result for inclusion in the conversation.
    
    Args:
        result: Dict with 'success', 'output', and 'error' keys
    
    Returns:
        Formatted string for the conversation
    """
    parts = []
    
    if result.get('output'):
        parts.append(f"**Output:**\n```\n{result['output']}\n```")
    
    if result.get('error'):
        if result.get('success'):
            parts.append(f"**Stderr:**\n```\n{result['error']}\n```")
        else:
            parts.append(f"**Error:**\n```\n{result['error']}\n```")
    
    if not parts:
        if result.get('success'):
            parts.append("*(Code executed successfully with no output)*")
        else:
            parts.append("*(Execution failed with no output)*")
    
    return '\n\n'.join(parts)


def clean_response_for_display(response: str, max_length: int = 500) -> str:
    """
    Clean and truncate a response for display purposes.
    
    Args:
        response: Raw response text
        max_length: Maximum display length
    
    Returns:
        Cleaned and possibly truncated response
    """
    # Remove excessive whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', response)
    cleaned = cleaned.strip()
    
    if len(cleaned) <= max_length:
        return cleaned
    
    return cleaned[:max_length] + "..."
