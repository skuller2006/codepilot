"""
Repository tools — safe file system operations for CodePilot AI.

ALL file access goes through this module.
Path traversal attacks are prevented by resolving paths and verifying
they remain within the repository root.

SECURITY: This module never executes any code from the repository.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

# File extensions treated as source code
SOURCE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rb", ".php",
    ".cs", ".cpp", ".c", ".h", ".rs", ".swift", ".kt", ".scala", ".sh",
    ".bash", ".zsh", ".yml", ".yaml", ".json", ".toml", ".tf", ".sql",
    ".html", ".css", ".vue", ".svelte",
}

# File extensions that should NOT be read (binary, compiled, etc.)
BINARY_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".dll", ".exe", ".bin", ".class",
    ".jar", ".war", ".zip", ".tar", ".gz", ".rar", ".7z",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".pdf",
    ".mp3", ".mp4", ".avi", ".mov", ".woff", ".woff2", ".ttf",
    ".eot", ".min.js",
}

# Max file size to read (bytes) — skip large files
MAX_FILE_SIZE = 500_000  # 500 KB


# ---------------------------------------------------------------------------
# Safe ZIP Extraction
# ---------------------------------------------------------------------------

def safe_extract_zip(zip_path: str, extract_to: str) -> Tuple[bool, str, str]:
    """
    Safely extract a ZIP archive, preventing path traversal attacks.

    Args:
        zip_path: Path to the uploaded ZIP file.
        extract_to: Target extraction directory.

    Returns:
        (success, extract_root, error_message)
    """
    extract_root = Path(extract_to)
    extract_root.mkdir(parents=True, exist_ok=True)
    resolved_root = extract_root.resolve()

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                # Reject absolute paths and traversal sequences
                if os.path.isabs(member):
                    return False, "", f"ZIP contains absolute path: {member}"
                if ".." in Path(member).parts:
                    return False, "", f"ZIP contains path traversal: {member}"

                # Resolve target and ensure it stays inside extract_root
                target = (resolved_root / member).resolve()
                if not str(target).startswith(str(resolved_root)):
                    return False, "", f"ZIP path escapes extraction dir: {member}"

            # Safe — extract
            zf.extractall(str(resolved_root))

    except zipfile.BadZipFile:
        return False, "", "Invalid or corrupted ZIP file."
    except Exception as exc:
        return False, "", f"Extraction error: {exc}"

    # If ZIP had a single top-level directory, use that as the repo root
    entries = list(resolved_root.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        return True, str(entries[0]), ""

    return True, str(resolved_root), ""


# ---------------------------------------------------------------------------
# Path Validation Helper
# ---------------------------------------------------------------------------

def _safe_path(repo_root: str, relative_path: str) -> Optional[Path]:
    """
    Resolve a relative path within the repository root.
    Returns None if the path would escape the root.
    """
    root = Path(repo_root).resolve()
    candidate = (root / relative_path).resolve()
    if not str(candidate).startswith(str(root)):
        return None
    return candidate


# ---------------------------------------------------------------------------
# Tool: repository_tree
# ---------------------------------------------------------------------------

def repository_tree(repo_root: str, max_depth: int = 4) -> str:
    """
    Return a human-readable directory tree of the repository.

    Args:
        repo_root: Absolute path to the repository root.
        max_depth: Maximum directory depth to display.

    Returns:
        Multi-line string representing the tree.
    """
    root = Path(repo_root)
    lines: List[str] = [root.name + "/"]

    def _walk(path: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda e: (e.is_file(), e.name))
        except PermissionError:
            return

        # Skip hidden and common noise directories
        entries = [
            e for e in entries
            if not e.name.startswith(".")
            and e.name not in {"__pycache__", "node_modules", ".git", "venv", ".venv", "dist", "build"}
        ]

        for i, entry in enumerate(entries):
            connector = "└── " if i == len(entries) - 1 else "├── "
            lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
            if entry.is_dir():
                extension = "    " if i == len(entries) - 1 else "│   "
                _walk(entry, prefix + extension, depth + 1)

    _walk(root, "", 1)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: read_file
# ---------------------------------------------------------------------------

def read_file(repo_root: str, relative_path: str) -> Tuple[bool, str]:
    """
    Safely read a file from within the repository.

    Args:
        repo_root: Absolute path to the repository root.
        relative_path: Path relative to the repository root.

    Returns:
        (success, content_or_error_message)
    """
    safe = _safe_path(repo_root, relative_path)
    if safe is None:
        return False, f"Path traversal detected: {relative_path}"

    if not safe.exists():
        return False, f"File not found: {relative_path}"

    if not safe.is_file():
        return False, f"Not a file: {relative_path}"

    # Check extension
    suffix = safe.suffix.lower()
    if suffix in BINARY_EXTENSIONS or suffix == "":
        # Try for common no-extension config files
        if safe.name not in {
            "Dockerfile", "Makefile", "Jenkinsfile", "Procfile",
            "README", "LICENSE", "CHANGELOG", ".env.example",
        }:
            return False, f"Binary or unsupported file type: {relative_path}"

    # Check size
    size = safe.stat().st_size
    if size > MAX_FILE_SIZE:
        return True, (
            f"[File truncated — {size:,} bytes > {MAX_FILE_SIZE:,} limit]\n"
            + safe.read_text(encoding="utf-8", errors="replace")[:MAX_FILE_SIZE]
        )

    try:
        content = safe.read_text(encoding="utf-8", errors="replace")
        return True, content
    except Exception as exc:
        return False, f"Error reading file: {exc}"


# ---------------------------------------------------------------------------
# Tool: find_files
# ---------------------------------------------------------------------------

def find_files(
    repo_root: str,
    extension: Optional[str] = None,
    name_pattern: Optional[str] = None,
    max_results: int = 200,
) -> List[str]:
    """
    Find files in the repository matching an extension or name pattern.

    Args:
        repo_root: Repository root path.
        extension: File extension (e.g. ".py"), or None for all source files.
        name_pattern: Exact filename (e.g. "requirements.txt").
        max_results: Maximum number of results.

    Returns:
        List of relative file paths.
    """
    root = Path(repo_root).resolve()
    results: List[str] = []

    for path in root.rglob("*"):
        if len(results) >= max_results:
            break
        if not path.is_file():
            continue
        # Skip hidden / build dirs
        parts = path.relative_to(root).parts
        if any(p.startswith(".") or p in {"__pycache__", "node_modules", "venv", ".venv", "dist", "build"} for p in parts):
            continue

        if name_pattern and path.name == name_pattern:
            results.append(str(path.relative_to(root)))
        elif extension and path.suffix.lower() == extension.lower():
            results.append(str(path.relative_to(root)))
        elif not extension and not name_pattern:
            if path.suffix.lower() in SOURCE_EXTENSIONS:
                results.append(str(path.relative_to(root)))

    return results


# ---------------------------------------------------------------------------
# Tool: list_directory
# ---------------------------------------------------------------------------

def list_directory(repo_root: str, relative_path: str = "") -> List[Dict]:
    """
    List the contents of a directory within the repository.
    """
    safe = _safe_path(repo_root, relative_path) if relative_path else Path(repo_root).resolve()
    if safe is None or not safe.is_dir():
        return []

    entries = []
    for entry in sorted(safe.iterdir(), key=lambda e: (e.is_file(), e.name)):
        entries.append({
            "name": entry.name,
            "type": "directory" if entry.is_dir() else "file",
            "size": entry.stat().st_size if entry.is_file() else None,
            "extension": entry.suffix.lower() if entry.is_file() else None,
        })
    return entries


# ---------------------------------------------------------------------------
# File Metadata
# ---------------------------------------------------------------------------

def get_repo_metadata(repo_root: str) -> Dict:
    """
    Gather basic metadata about the repository without sending to LLM.
    Pure deterministic analysis.
    """
    root = Path(repo_root).resolve()
    all_files = find_files(repo_root)

    ext_counts: Dict[str, int] = {}
    for f in all_files:
        ext = Path(f).suffix.lower()
        if ext:
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

    # Detect test directories
    test_dirs = []
    for name in ["tests", "test", "spec", "__tests__", "e2e"]:
        if (root / name).is_dir():
            test_dirs.append(name)

    # Check for common config files
    config_files = []
    for cfg in [
        "requirements.txt", "pyproject.toml", "setup.py", "package.json",
        "pom.xml", "build.gradle", "Cargo.toml", "go.mod", "Gemfile",
        "docker-compose.yml", "docker-compose.yaml", ".env.example",
        "Dockerfile", ".github", "Makefile", "tox.ini", "pytest.ini",
        ".eslintrc.json", ".eslintrc.js", "tsconfig.json",
    ]:
        if (root / cfg).exists():
            config_files.append(cfg)

    # Check for CI files
    has_ci = (
        (root / ".github" / "workflows").is_dir()
        or (root / ".gitlab-ci.yml").exists()
        or (root / "Jenkinsfile").exists()
        or (root / ".circleci").is_dir()
    )

    # README
    readme_files = []
    for rd in ["README.md", "README.rst", "README.txt", "README"]:
        if (root / rd).exists():
            readme_files.append(rd)

    return {
        "total_files": len(all_files),
        "extension_counts": ext_counts,
        "test_directories": test_dirs,
        "config_files": config_files,
        "has_ci": has_ci,
        "readme_files": readme_files,
        "root_path": str(root),
    }
