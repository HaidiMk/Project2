"""
Adds the AI project's folders to sys.path so Django code can import from them
(Expert System/, TOPSIS/, Nlp/) regardless of the working directory manage.py
is launched from. Anchored to this file's own location so it keeps working if
the repo is ever moved or renamed.

Import this module once, early, from settings.py (see the import line added
at the top of settings.py).
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]   # .../<repo>/Backend/config/ai_bridge.py -> repo root
_PATHS_TO_ADD = [
    _REPO_ROOT,                          # for `import Nlp`, `import Nlp.pipeline`
    _REPO_ROOT / "Expert System",        # for `from core...`, `from rules...`, `from engine...`, `from ml...`
    _REPO_ROOT / "TOPSIS",               # for `import topsis_model` (used internally by meal_planner)
]

for _p in _PATHS_TO_ADD:
    _p_str = str(_p)
    if _p_str in sys.path:
        continue
    if _p.exists():
        sys.path.append(_p_str)
    else:
        sys.stderr.write(
            f"WARNING: ai_bridge.py expected AI project folder '{_p.name}' at {_p_str}, "
            "but it does not exist; imports of that project will fail.\n"
        )