"""
RLM Engine - Recursive Language Model CLI.

Usage:
    python main.py "Your query here" path/to/document.txt
    python main.py --help
"""

import os
import sys
import argparse
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from rlm.agent import RLMAgent


def main():
    parser = argparse.ArgumentParser(
        description="RLM Engine - Analyze large documents using recursive LLM queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py "Summarize the main themes" book.txt
    python main.py "Find all mentions of project deadlines" report.txt --type "project report"
    python main.py "What are the key findings?" research_paper.txt --max-turns 20
        """
    )
    
    parser.add_argument(
        "query",
        type=str,
        help="The question or task to perform on the document"
    )
    
    parser.add_argument(
        "file",
        type=str,
        help="Path to the context file (text document)"
    )
    
    parser.add_argument(
        "--type", "-t",
        type=str,
        default="text document",
        help="Type of document (e.g., 'book', 'research paper', 'code')"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="xiaomi/mimo-v2-flash:free",
        help="OpenRouter model to use for root agent"
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
        help="Only build the Docker image, don't run a query"
    )
    
    args = parser.parse_args()
    
    # Validate file exists
    if not args.build_only:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        
        if not file_path.is_file():
            print(f"Error: Not a file: {args.file}", file=sys.stderr)
            sys.exit(1)
    
    # Check API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set", file=sys.stderr)
        print("Set it in your .env file or export it in your shell", file=sys.stderr)
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
    
    # Create and run agent
    agent = RLMAgent(
        api_key=api_key,
        model=args.model,
        max_turns=args.max_turns,
        verbose=not args.quiet
    )
    
    try:
        result = agent.run(
            query=args.query,
            context_file=str(file_path.absolute()),
            context_type=args.type
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
        sys.exit(1)


if __name__ == "__main__":
    main()
