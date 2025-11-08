#!/usr/bin/env python3
"""
Startup script for Claude Code Proxy
"""
import os
import sys

if __name__ == "__main__":
    # Add src directory to Python path
    src_dir = os.path.join(os.path.dirname(__file__), "src")
    if os.path.exists(src_dir):
        sys.path.insert(0, src_dir)
    
    # Import and run
    try:
        import uvicorn
        from main import app, HOST, PORT
        
        uvicorn.run(app, host=HOST, port=PORT)
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install dependencies first:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
