import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]   
_PATHS_TO_ADD = [
    _REPO_ROOT,                         
    _REPO_ROOT / "Expert System",      
    _REPO_ROOT / "TOPSIS",           
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