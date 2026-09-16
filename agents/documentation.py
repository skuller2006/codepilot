"""
Documentation Agent — CodePilot AI.

Checks README quality, setup instructions, API documentation,
module docstrings, and inline documentation.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List

from config import get_llm
from graph.state import AnalysisState, AgentTraceEntry
from models.schemas import Finding, FindingCategory, FindingSource, Severity
from agents.base import (
    build_code_context,
    get_reanalysis_context,
    parse_findings_from_llm,
    read_files_for_analysis,
)
from tools.repository import find_files, read_file

logger = logging.getLogger(__name__)

DOC_PROMPT = """You are a technical documentation expert reviewing a codebase.

{critic_feedback}

Repository context:
- Languages: {languages}
- Frameworks: {frameworks}
- Focus: {focus_areas}

README content:
{readme_content}

Sample source files (check docstrings):
{code_context}

Documentation audit:
{doc_audit}

Identify documentation gaps and quality issues:
1. README missing installation instructions
2. README missing usage examples
3. README missing configuration documentation
4. Public functions/classes missing docstrings
5. API endpoints missing documentation
6. Complex logic without inline comments
7. Missing type hints (for Python)
8. Missing changelog or versioning information
9. Unclear or absent contribution guidelines
10. Configuration options not documented

Return a JSON array:
[
  {{
    "title": "Specific documentation gap",
    "severity": "Medium|Low|Info",
    "file": "path/to/file.py",
    "line": null,
    "description": "What documentation is missing and why it matters",
    "evidence": "The undocumented code or the gap in the README",
    "recommendation": "What documentation should be added and how",
    "confidence": 0.85
  }}
]

Return ONLY the JSON array.
"""


def _audit_documentation(repo_root: str) -> tuple[str, List[Finding]]:
    """
    Deterministic documentation audit.
    Returns (audit_text, findings).
    """
    findings: List[Finding] = []
    audit_parts = []

    root = Path(repo_root).resolve()

    # Check README existence and length
    readme_path = None
    readme_content = ""
    for rd in ["README.md", "README.rst", "README.txt", "README"]:
        p = root / rd
        if p.exists():
            readme_path = rd
            ok, content = read_file(repo_root, rd)
            if ok:
                readme_content = content
            break

    if not readme_path:
        audit_parts.append("❌ No README file found")
        findings.append(Finding(
            category=FindingCategory.DOCUMENTATION,
            title="No README file",
            severity=Severity.HIGH,
            file=None,
            line=None,
            description="The repository has no README file, making it impossible for users to understand the project's purpose, how to install it, or how to use it.",
            evidence="No README.md, README.rst, or README.txt found in root",
            recommendation="Create a comprehensive README.md with: project description, requirements, installation, usage examples, and contribution guidelines.",
            confidence=1.0,
            source=FindingSource.DETERMINISTIC,
        ))
    else:
        readme_lines = len(readme_content.splitlines())
        audit_parts.append(f"✓ README found: {readme_path} ({readme_lines} lines)")

        if readme_lines < 20:
            audit_parts.append("⚠ README is very short (< 20 lines)")
            findings.append(Finding(
                category=FindingCategory.DOCUMENTATION,
                title="README is insufficient",
                severity=Severity.MEDIUM,
                file=readme_path,
                line=None,
                description=f"The README is only {readme_lines} lines, which is insufficient for users or contributors to understand the project.",
                evidence=f"README.md: {readme_lines} lines total",
                recommendation="Expand the README to include: overview, prerequisites, installation, configuration, usage examples, and contributing guidelines.",
                confidence=0.95,
                source=FindingSource.DETERMINISTIC,
            ))

        # Check for common sections
        readme_lower = readme_content.lower()
        for section in ["installation", "usage", "configuration", "requirements", "example"]:
            if section not in readme_lower:
                audit_parts.append(f"⚠ README missing '{section}' section")

    # Check for missing docstrings in Python files
    py_files = find_files(repo_root, extension=".py")
    undocumented_count = 0
    for rel_path in py_files[:20]:
        if "test" in rel_path.lower():
            continue
        ok, content = read_file(repo_root, rel_path)
        if not ok:
            continue
        # Check if functions lack docstrings
        functions = re.findall(r"def (\w+)\s*\([^)]*\)\s*:", content)
        for func_name in functions:
            # Find position after the def line
            match = re.search(r"def " + re.escape(func_name) + r"\s*\([^)]*\)\s*:(.*?)(?:def|\Z)", content, re.DOTALL)
            if match:
                body = match.group(1).strip()
                if not body.startswith('"""') and not body.startswith("'''"):
                    undocumented_count += 1

    if undocumented_count > 5:
        audit_parts.append(f"⚠ {undocumented_count} functions/methods appear to lack docstrings")
        findings.append(Finding(
            category=FindingCategory.DOCUMENTATION,
            title=f"Missing docstrings: {undocumented_count} undocumented functions",
            severity=Severity.LOW,
            file=None,
            line=None,
            description=f"Approximately {undocumented_count} functions/methods in the codebase lack docstrings, reducing code maintainability and making IDE assistance less effective.",
            evidence=f"Scanned {min(20, len(py_files))} Python files, found ~{undocumented_count} undocumented functions",
            recommendation="Add docstrings to all public functions and classes following the Google or NumPy docstring convention.",
            confidence=0.80,
            source=FindingSource.DETERMINISTIC,
        ))

    # Check for changelog
    if not (root / "CHANGELOG.md").exists() and not (root / "CHANGELOG.rst").exists():
        audit_parts.append("⚠ No CHANGELOG file found")

    return "\n".join(audit_parts), findings


def run_documentation_agent(state: AnalysisState) -> Dict:
    """
    LangGraph node: Documentation Agent.

    Args:
        state: Current graph state.

    Returns:
        State updates dict.
    """
    repo_root = state["repo_path"]
    revision_count = state.get("revision_count", 0)
    logger.info(f"[Documentation] Running (revision {revision_count})")

    reanalysis_files, critic_feedback = get_reanalysis_context(state, "documentation")

    # Run deterministic audit
    doc_audit, det_findings = _audit_documentation(repo_root)

    # Read README
    readme_content = "No README found."
    for rd in ["README.md", "README.rst", "README.txt"]:
        ok, content = read_file(repo_root, rd)
        if ok:
            readme_content = content[:4000]
            break

    # Select some source files to check docstrings
    files = find_files(repo_root, extension=".py", max_results=10)
    files = [f for f in files if "test" not in f.lower()][:6]
    if reanalysis_files:
        files = reanalysis_files[:6]

    file_contents = read_files_for_analysis(repo_root, files)
    code_context = build_code_context(file_contents, max_total_chars=8000)

    repo_summary = state.get("repo_summary")
    languages = ", ".join(repo_summary.languages) if repo_summary else "Unknown"
    frameworks = ", ".join(repo_summary.frameworks) if repo_summary else "Unknown"
    plan = state.get("investigation_plan")
    focus_areas = ", ".join(plan.focus_areas) if plan else "documentation"

    prompt = DOC_PROMPT.format(
        critic_feedback=critic_feedback or "",
        languages=languages,
        frameworks=frameworks,
        focus_areas=focus_areas,
        readme_content=readme_content,
        code_context=code_context,
        doc_audit=doc_audit,
    )

    llm_findings = []
    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        llm_findings = parse_findings_from_llm(
            response.content,
            FindingCategory.DOCUMENTATION,
            "Documentation",
        )
    except Exception as exc:
        logger.error(f"[Documentation] LLM error: {exc}", exc_info=True)

    all_findings = det_findings + llm_findings
    summary = f"Documentation review: {len(all_findings)} gaps identified."

    trace_entry: AgentTraceEntry = {
        "node": "Documentation Agent",
        "iteration": revision_count + 1,
        "action": "Reviewed README, docstrings, and API documentation",
        "result": f"Found {len(all_findings)} documentation gaps ({len(det_findings)} deterministic)",
    }

    logger.info(f"[Documentation] Complete: {len(all_findings)} findings")

    return {
        "all_findings": all_findings,
        "documentation_summary": summary,
        "documentation_files": files,
        "agent_trace": [trace_entry],
    }
