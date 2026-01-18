#!/usr/bin/env python3
"""
RLM Test Script - Verify the implementation works correctly.

Run this script to test the RLM implementation:
    python tests/test_rlm.py

Prerequisites:
    1. Docker must be installed and running
    2. OPENROUTER_API_KEY must be set in .env or environment
    3. Build the Docker image first: docker build -t rlm-sandbox .
"""

import os
import sys
import time

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def test_imports():
    """Test that all modules can be imported."""
    print("=" * 60)
    print("TEST: Module Imports")
    print("=" * 60)
    
    try:
        from rlm.agent import RLMAgent
        print("✓ RLMAgent imported")
        
        from rlm.clients.docker_sandbox import DockerSandbox
        print("✓ DockerSandbox imported")
        
        from rlm.clients.remote_sandbox import RemoteSandbox
        print("✓ RemoteSandbox imported")
        
        from rlm.clients.self_sandbox import SelfSandbox
        print("✓ SelfSandbox imported")
        
        from rlm.prompts import format_system_prompt
        print("✓ prompts imported")
        
        print("\n All imports successful!\n")
        return True
    except Exception as e:
        print(f"\n Import failed: {e}\n")
        return False


def test_self_sandbox():
    """Test the SelfSandbox in-process execution."""
    print("=" * 60)
    print("TEST: SelfSandbox In-Process Execution")
    print("=" * 60)
    
    from rlm.clients.self_sandbox import SelfSandbox
    
    # Create sandbox with empty namespace
    namespace = {}
    file_index = {"test.txt": {"size": 100, "type": "text"}}
    
    def mock_list_files(pattern="*"):
        return ["test.txt", "main.py"]
    
    def mock_read_file(path):
        return f"Content of {path}"
    
    sandbox = SelfSandbox(
        namespace=namespace,
        file_index=file_index,
        helper_functions={
            'list_files': mock_list_files,
            'read_file': mock_read_file,
        }
    )
    
    # Test 1: Basic execution
    result = sandbox.exec_code("x = 1 + 1\nprint(f'Result: {x}')")
    assert result['success'], f"Execution failed: {result['error']}"
    assert 'Result: 2' in result['output'], f"Wrong output: {result['output']}"
    print("✓ Basic code execution works")
    
    # Test 2: Helper functions available
    result = sandbox.exec_code("files = list_files('*')\nprint(files)")
    assert result['success'], f"Helper function failed: {result['error']}"
    assert 'test.txt' in result['output'], f"Helper not working: {result['output']}"
    print("✓ Helper functions injected correctly")
    
    # Test 3: Variable persistence
    sandbox.exec_code("my_var = 42")
    value = sandbox.get_variable("my_var")
    assert value == 42, f"Variable not persisted: {value}"
    print("✓ Variable persistence works")
    
    print("\nSelfSandbox tests passed!\n")
    return True


def test_docker_sandbox_build():
    """Test that Docker sandbox can build the image."""
    print("=" * 60)
    print("TEST: Docker Image Build")
    print("=" * 60)
    
    from rlm.clients.docker_sandbox import DockerSandbox
    
    sandbox = DockerSandbox(verbose=True)
    
    # Check if image exists or build it
    import subprocess
    result = subprocess.run(
        ["docker", "images", "-q", "rlm-sandbox"],
        capture_output=True,
        text=True
    )
    
    if result.stdout.strip():
        print("✓ Docker image already exists")
    else:
        print("Building Docker image (this may take a minute)...")
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        success = sandbox.build_image(project_dir)
        if success:
            print("✓ Docker image built successfully")
        else:
            print(" Failed to build Docker image")
            return False
    
    print("\n Docker image ready!\n")
    return True


def test_docker_sandbox_start():
    """Test that Docker sandbox can start and respond."""
    print("=" * 60)
    print("TEST: Docker Sandbox Start & HTTP Communication")
    print("=" * 60)
    
    from rlm.clients.docker_sandbox import DockerSandbox
    
    # Use the rlm directory as test data
    test_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "rlm"
    )
    
    sandbox = DockerSandbox(
        context_file=test_dir,
        verbose=True
    )
    
    try:
        print("Starting container...")
        if not sandbox.start():
            print(" Failed to start container")
            return False
        print("✓ Container started")
        
        # Test ping
        if sandbox.ping():
            print("✓ HTTP ping successful")
        else:
            print(" HTTP ping failed")
            return False
        
        # Test code execution
        result = sandbox.exec_code("print('Hello from sandbox!')")
        if result['success'] and 'Hello from sandbox' in result['output']:
            print("✓ Code execution works")
        else:
            print(f" Code execution failed: {result}")
            return False
        
        # Test list_files
        result = sandbox.exec_code("files = list_files('*.py')\nprint(files)")
        if result['success']:
            print(f"✓ list_files() works: {result['output'][:100]}...")
        else:
            print(f" list_files failed: {result['error']}")
            return False
        
        print("\n Docker sandbox tests passed!\n")
        return True
        
    finally:
        print("Stopping container...")
        sandbox.stop()


def test_full_agent():
    """Test a full agent run (requires API key and Docker)."""
    print("=" * 60)
    print("TEST: Full Agent Run (E2E)")
    print("=" * 60)
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("  Skipping: OPENROUTER_API_KEY not set")
        return True
    
    from rlm.agent import RLMAgent
    
    agent = RLMAgent(
        api_key=api_key,
        max_turns=3,
        verbose=True
    )
    
    # Simple test: count Python files in rlm directory
    test_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "rlm"
    )
    
    try:
        result = agent.run_directory(
            query="List all Python files and count them",
            directory=test_dir,
            context_type="Python package"
        )
        
        print(f"\nAgent result:\n{result[:500]}")
        print("\n Full agent run completed!\n")
        return True
        
    except Exception as e:
        print(f" Agent run failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RLM IMPLEMENTATION TESTS")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1: Imports
    results.append(("Imports", test_imports()))
    
    # Test 2: SelfSandbox
    results.append(("SelfSandbox", test_self_sandbox()))
    
    # Test 3: Docker build (optional - skip if no Docker)
    try:
        import subprocess
        subprocess.run(["docker", "--version"], capture_output=True, check=True)
        results.append(("Docker Build", test_docker_sandbox_build()))
        results.append(("Docker Start", test_docker_sandbox_start()))
    except Exception:
        print("  Docker not available, skipping Docker tests\n")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = " PASS" if passed else " FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! ")
    else:
        print("Some tests failed. See above for details.")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
