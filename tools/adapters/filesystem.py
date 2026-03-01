"""Restricted filesystem read/write tool adapter.

All paths are resolved relative to a base directory and cannot escape it.
"""

from pathlib import Path


def _validate_path(path: str, base_dir: str) -> Path:
    base = Path(base_dir).resolve()
    target = (base / path).resolve()
    if not str(target).startswith(str(base)):
        raise PermissionError(f"Path escapes base directory: {path}")
    return target


def read_file(path: str, base_dir: str = "data") -> dict:
    target = _validate_path(path, base_dir)
    if not target.exists():
        raise FileNotFoundError(f"File not found: {path}")
    content = target.read_text(encoding="utf-8", errors="replace")
    return {"path": str(target), "content": content, "size": len(content)}


def write_file(path: str, content: str, base_dir: str = "data") -> dict:
    target = _validate_path(path, base_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"path": str(target), "written": len(content)}


def list_directory(path: str = ".", base_dir: str = "data") -> dict:
    target = _validate_path(path, base_dir)
    if not target.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    entries = []
    for entry in sorted(target.iterdir()):
        entries.append({
            "name": entry.name,
            "type": "dir" if entry.is_dir() else "file",
            "size": entry.stat().st_size if entry.is_file() else None,
        })
    return {"path": str(target), "entries": entries}
