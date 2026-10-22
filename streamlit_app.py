"""
Streamlit Community Cloud Entry Point.
Delegates execution directly to src/ui_app.py.
"""
import os
import sys
import runpy

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

target_script = os.path.join(PROJECT_ROOT, "src", "ui_app.py")
runpy.run_path(target_script, run_name="__main__")
