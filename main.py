#!/usr/bin/env python3
"""
RLM Engine - Recursive Language Model CLI.

Usage:
    python main.py "Your query" path/to/file.txt
    python main.py "Your query" --directory path/to/folder
    python main.py "Your query" --remote http://server:8080 --directory /data
"""

import os
import sys
import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from rlm.agent import RLMAgent


def main():
    parser = argparse.ArgumentParser(
        description="RLM Engine - Analyze documents/directories using recursive LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single file analysis (local)
    python main.py "Summarize the main themes" book.txt
    
    # Directory analysis (requires remote server)
    python main.py "Explain this codebase architecture" --directory ./my_project \\
        --remote http://localhost:8080
    
    # Remote server with API key
    python main.py "Find security issues" --directory ./src \\
        --remote https://rlm.example.com:8080 --remote-key your-api-key
        """
    )
    
    parser.add_argument(
        "query",
        type=str,
        help="The question or task to perform"
    )
    
    parser.add_argument(
        "file",
        type=str,
        nargs="?",
        default=None,
        help="Path to context file (for single-file mode)"
    )
    
    parser.add_argument(
        "--directory", "-d",
        type=str,
        default=None,
        help="Path to directory to analyze (directory mode)"
    )
    
    parser.add_argument(
        "--remote", "-r",
        type=str,
        default=None,
        help="URL of remote sandbox server (e.g., http://localhost:8080)"
    )
    
    parser.add_argument(
        "--remote-key",
        type=str,
        default=None,
        help="API key for remote sandbox authentication"
    )
    
    parser.add_argument(
        "--type", "-t",
        type=str,
        default=None,
        help="Type of content (e.g., 'codebase', 'book', 'research paper')"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="xiaomi/mimo-v2-flash:free",
        help="OpenRouter model to use"
    )
    
    parser.add_argument(
        "--max-turns",
        type=int,
        default=15,
        help="Maximum conversation turns (default: 15)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only build the Docker image, don't run"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.build_only:
        if args.directory is None and args.file is None:
            print("Error: Provide either a file path or --directory", file=sys.stderr)
            sys.exit(1)
        
        if args.directory and args.file:
            print("Error: Provide either file or --directory, not both", file=sys.stderr)
            sys.exit(1)
    
    # Check API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    
    # Build-only mode
    if args.build_only:
        from rlm.clients.docker_sandbox import DockerSandbox
        sandbox = DockerSandbox()
        print("Building Docker image...")
        if sandbox.build_image("."):
            print("Docker image built successfully!")
            sys.exit(0)
        else:
            print("Failed to build Docker image", file=sys.stderr)
            sys.exit(1)
    
    # Create agent
    agent = RLMAgent(
        api_key=api_key,
        model=args.model,
        max_turns=args.max_turns,
        verbose=not args.quiet,
        remote_url=args.remote,
        remote_api_key=args.remote_key
    )
    
    try:
        if args.directory:
            # Directory mode
            directory = Path(args.directory)
            if not directory.exists():
                print(f"Error: Directory not found: {args.directory}", file=sys.stderr)
                sys.exit(1)
            
            context_type = args.type or "codebase"
            result = agent.run_directory(
                query=args.query,
                directory=str(directory.absolute()),
                context_type=context_type
            )
        else:
            # File mode
            file_path = Path(args.file)
            if not file_path.exists():
                print(f"Error: File not found: {args.file}", file=sys.stderr)
                sys.exit(1)
            
            context_type = args.type or "text document"
            result = agent.run(
                query=args.query,
                context_file=str(file_path.absolute()),
                context_type=context_type
            )
        
        print("\n" + "="*60)
        print("FINAL ANSWER:")
        print("="*60)
        print(result)
        print("="*60)
        
        if not args.quiet:
            print(f"\nCompleted in {agent.get_turn_count()} turns")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
