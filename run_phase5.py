#!/usr/bin/env python3
"""
CLI Entrypoint for Real-Time Subtitle Segmenter and N-gram Phrase Extractor (Phase 5).

Launches the local Streamlit Web UI dashboard.

Usage:
    python run_phase5.py
    python run_phase5.py --port 8501
    python run_phase5.py --port 8501 --headless
"""

import os
import sys
import argparse
import subprocess

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Enforce project-local NLTK data directory
NLTK_DATA_DIR = os.path.join(PROJECT_ROOT, "nltk_data")
if os.path.exists(NLTK_DATA_DIR):
    import nltk
    if NLTK_DATA_DIR not in nltk.data.path:
        nltk.data.path.insert(0, NLTK_DATA_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: Real-Time Subtitle Segmenter & Phrase Extractor Web UI"
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8501,
        help="Port to bind Streamlit web server to (default: 8501)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Streamlit in headless mode (without automatically opening a browser)",
    )

    args = parser.parse_args()

    app_path = os.path.join(PROJECT_ROOT, "src", "ui_app.py")
    if not os.path.exists(app_path):
        print(f"Error: UI application script not found at {app_path}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.port",
        str(args.port),
        "--server.headless",
        "true" if args.headless else "false",
        "--theme.base",
        "light",
    ]

    print("=" * 70)
    print("STARTING PHASE 5 REAL-TIME SUBTITLE WEB UI")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"App Script:   {app_path}")
    print(f"Local URL:    http://localhost:{args.port}")
    print("=" * 70)

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[INFO] Phase 5 Web UI stopped by user.")
    except Exception as e:
        print(f"\n[ERROR] Failed to start Streamlit: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
