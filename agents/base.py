"""
Base utilities shared by all specialist agents in CodePilot AI.

Provides:
- File selection logic (respects investigation plan)
- LLM finding parsing helpers
- Standard trace entry creation
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from config import MAX_FILES_PER_AGENT, MAX_TOKENS_PER_FILE, get_llm
from graph.state import AnalysisState, AgentTraceEntry
from models.schemas import Finding, FindingCategory, FindingSource, Severity
from tools.repository import find_files, read_file

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File Selection
# ---------------------------------------------------------------------------

def select_files_for_agent(
    state: AnalysisState,
    preferred_extensions: List[str],
    max_files: int = MAX_FILES_PER_AGENT,
    reanalysis_files: Optional[List[str]] = None,
) -> List[str]:
    """
    Select files for an agent to analyze.

    Priority order:
    1. Files specified in reanalysis request
    2. Key files from investigation plan
    3. Files matching preferred extensions

    Args:
        state: Current analysis state.
        preferred_extensions: Extensions to look for (e.g. [".py"]).
        max_files: Maximum number of files to return.
        reanalysis_files: Specific files requested by critic.

    Returns:
        List of relative file paths.
    """
    repo_root = state["repo_path"]
    selected = []

    # Priority 1: Critic-requested reanalysis files
    if reanalysis_files:
        selected.extend(reanalysis_files[:max_files])
        if len(selected) >= max_files:
            return selected[:max_files]

    # Priority 2: Key files from investigation plan
    plan = state.get("investigation_plan")
    if plan and plan.key_files_to_examine:
        for f in plan.key_files_to_examine:
            if f not in selected:
                selected.append(f)

    # Priority 3: Files by extension
    remaining = max_files - len(selected)
    for ext in preferred_extensions:
        if remaining <= 0:
            break
        ext_files = find_files(repo_root, extension=ext, max_results=remaining * 2)
        for f in ext_files:
            if f not in selected and len(selected) < max_files:
                selected.append(f)
        remaining = max_files - len(selected)

    return selected[:max_files]


def read_files_for_analysis(
    repo_root: str,
    file_paths: List[str],
    max_chars_per_file: int = MAX_TOKENS_PER_FILE,
) -> List[Dict]:
    """
    Read multiple files, returning their content truncated to max_chars_per_file.

    Returns:
        List of {"path": str, "content": str} dicts.
    """
    results = []
    for path in file_paths:
        ok, content = read_file(repo_root, path)
        if ok:
            results.append({
                "path": path,
                "content": content[:max_chars_per_file],
            })
    return results


def build_code_context(files: List[Dict], max_total_chars: int = 20000) -> str:
    """
    Build a formatted code context string from file contents.
    Truncates to max_total_chars total.
    """
    parts = []
    total = 0
    for f in files:
        header = f"\n\n=== FILE: {f['path']} ===\n"
        content = f['content']
        chunk = header + content
        if total + len(chunk) > max_total_chars:
            remaining = max_total_chars - total - len(header)
            if remaining > 200:
                parts.append(header + content[:remaining] + "\n... [truncated]")
            break
        parts.append(chunk)
        total += len(chunk)
    return "".join(parts)


# ---------------------------------------------------------------------------
# LLM Finding Parser
# ---------------------------------------------------------------------------

def parse_findings_from_llm(
    response_text: str,
    category: FindingCategory,
    agent_name: str,
) -> List[Finding]:
    """
    Parse structured findings from LLM response.

    Attempts JSON parse first, falls back to text parsing.
    """
    findings = []

    # Try to extract JSON array from the response
    json_match = re.search(r"\[[\s\S]*\]", response_text)
    if json_match:
        try:
            raw_findings = json.loads(json_match.group(0))
            for raw in raw_findings:
                try:
                    severity_str = raw.get("severity", "Medium")
                    # Normalize severity
                    severity_map = {s.value.lower(): s for s in Severity}
                    severity = severity_map.get(severity_str.lower(), Severity.MEDIUM)

                    finding = Finding(
                        category=category,
                        title=raw.get("title", "Unnamed Finding"),
                        severity=severity,
                        file=raw.get("file") or raw.get("path"),
                        line=raw.get("line") or raw.get("line_number"),
                        description=raw.get("description") or raw.get("explanation", ""),
                        evidence=raw.get("evidence") or raw.get("code_snippet"),
                        recommendation=raw.get("recommendation", "Review and fix."),
                        confidence=float(raw.get("confidence", 0.75)),
                        source=FindingSource.LLM,
                    )
                    findings.append(finding)
                except Exception as exc:
                    logger.debug(f"[{agent_name}] Skipped malformed finding: {exc}")
            return findings
        except json.JSONDecodeError:
            pass

    logger.debug(f"[{agent_name}] JSON parse failed, falling back to text parsing")
    return findings


def get_reanalysis_context(
    state: AnalysisState,
    agent_name: str,
) -> Tuple[List[str], str]:
    """
    Get reanalysis files and context for an agent re-running after critic rejection.

    Returns:
        (specific_files, critic_feedback_text)
    """
    reanalysis_files: List[str] = []
    critic_feedback = ""

    requests = state.get("reanalysis_requests", [])
    for req in requests:
        if req.agent_name.lower().replace(" ", "_") == agent_name.lower():
            reanalysis_files = req.specific_files
            critic_feedback = (
                f"CRITIC FEEDBACK: {req.reason}\n"
                f"Focus on: {', '.join(req.focus_areas)}"
            )
            break

    return reanalysis_files, critic_feedback
