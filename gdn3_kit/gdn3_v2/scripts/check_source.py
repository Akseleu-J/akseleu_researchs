"""Parse source without importing or executing the project or its tests."""
import ast
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
errors=[];count=0
for path in sorted(root.rglob("*.py")):
    if any(part in {".venv","__pycache__"} for part in path.parts):continue
    count+=1
    try:ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
    except SyntaxError as e:errors.append(f"{path.relative_to(root)}:{e.lineno}: {e.msg}")
print(f"Parsed {count} Python files; syntax errors: {len(errors)}")
for error in errors:print(error)
sys.exit(bool(errors))
