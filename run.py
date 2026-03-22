"""SAMpler launcher — python run.py"""
import os
import sys
from pathlib import Path

# Must be set before any torch/ultralytics import
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

def main():
    backend = Path(__file__).parent / "backend"
    frontend = Path(__file__).parent / "frontend"
    if not backend.exists():
        print(f"ERROR: backend/ directory not found at {backend}", file=sys.stderr)
        sys.exit(1)
    if not frontend.exists():
        print(f"ERROR: frontend/ directory not found at {frontend}", file=sys.stderr)
        sys.exit(1)

    import uvicorn
    print("SAMpler starting at http://localhost:8000")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
