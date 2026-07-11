"""Shared pytest setup.

`app.py` is a standalone Streamlit entrypoint at the repo root, not part of
the installed `agentic_analyst` package (which lives under src/ and is on
sys.path via the editable install). Add the repo root so `import app` works
from tests/test_app.py.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
