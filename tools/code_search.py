"""
Code search tool for CodePilot AI.

Provides pattern-based and text-based searching across repository source files.
All searches are read-only and deterministic — no LLM calls.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

from tools.repository import find_files, read_file, SOURCE_EXTENSIONS


# ---------------------------------------------------------------------------
# Tool: search_code
# ---------------------------------------------------------------------------

def search_code(
    repo_root: str,
    query: str,
    file_extensions: Optional[List[str]] = None,
    is_regex: bool = False,
    max_results: int = 50,
    context_lines: int = 2,
) -> List[Dict]:
    """
    Search source files for a text pattern or regex.

    Args:
        repo_root: Repository root path.
        query: Search term or regex pattern.
        file_extensions: Limit search to these extensions (e.g. [".py"]).
        is_regex: Treat query as a regex pattern if True.
        max_results: Maximum number of matches to return.
        context_lines: Number of context lines to include around each match.

    Returns:
        List of dicts with: file, line_number, line_content, context
    """
    results: List[Dict] = []
    root = Path(repo_root).resolve()

    # Build list of files to search
    extensions = file_extensions or list(SOURCE_EXTENSIONS)
    files_to_search: List[str] = []
    for ext in extensions:
        files_to_search.extend(find_files(repo_root, extension=ext))

    # Deduplicate
    files_to_search = list(dict.fromkeys(files_to_search))

    try:
        if is_regex:
            pattern = re.compile(query, re.IGNORECASE)
        else:
            pattern = re.compile(re.escape(query), re.IGNORECASE)
    except re.error as exc:
        return [{"error": f"Invalid regex pattern: {exc}"}]

    for rel_path in files_to_search:
        if len(results) >= max_results:
            break

        ok, content = read_file(repo_root, rel_path)
        if not ok:
            continue

        lines = content.splitlines()
        for i, line in enumerate(lines):
            if len(results) >= max_results:
                break
            if pattern.search(line):
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                context = "\n".join(
                    f"{'>>> ' if j == i else '    '}{lines[j]}"
                    for j in range(start, end)
                )
                results.append({
                    "file": rel_path,
                    "line_number": i + 1,
                    "line_content": line.strip(),
                    "context": context,
                })

    return results


def search_for_pattern_in_file(
    repo_root: str,
    relative_path: str,
    pattern: str,
    is_regex: bool = False,
) -> List[Dict]:
    """
    Search for a pattern within a single file.

    Returns:
        List of matches with line_number and line_content.
    """
    ok, content = read_file(repo_root, relative_path)
    if not ok:
        return []

    try:
        regex = re.compile(pattern if is_regex else re.escape(pattern), re.IGNORECASE)
    except re.error:
        return []

    results = []
    for i, line in enumerate(content.splitlines()):
        if regex.search(line):
            results.append({
                "line_number": i + 1,
                "line_content": line.strip(),
            })
    return results


def get_function_definitions(repo_root: str, relative_path: str) -> List[Dict]:
    """
    Extract function/method definitions from a Python file.
    Deterministic — no LLM.
    """
    ok, content = read_file(repo_root, relative_path)
    if not ok:
        return []

    results = []
    # Match Python function and class definitions
    pattern = re.compile(r"^(class|def)\s+(\w+)", re.MULTILINE)
    for match in pattern.finditer(content):
        line_no = content[: match.start()].count("\n") + 1
        results.append({
            "type": match.group(1),
            "name": match.group(2),
            "line_number": line_no,
        })
    return results
