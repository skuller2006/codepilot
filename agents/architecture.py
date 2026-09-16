"""
Architecture Agent — CodePilot AI.

Analyzes the repository structure and code organization to identify
architectural problems, coupling issues, and scalability concerns.
"""

from __future__ import annotations

import logging
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
    select_files_for_agent,
)
from tools.code_search import search_code
from tools.repository import find_files, repository_tree

logger = logging.getLogger(__name__)

ARCHITECTURE_PROMPT = """You are a software architect performing an architectural review.

{critic_feedback}

Repository context:
- Languages: {languages}
- Frameworks: {frameworks}
- Focus: {focus_areas}

Repository structure:
{directory_tree}

Key source files:
{code_context}

Structural metrics:
{metrics}

Analyze the ARCHITECTURE and identify:
1. Tight coupling — components that depend too heavily on each other's internals
2. Poor separation of concerns — business logic mixed with infrastructure/UI
3. God objects/modules — single files/classes doing too much
4. Missing abstraction layers — direct DB calls from controllers, etc.
5. Circular dependencies — module A imports B which imports A
6. Inconsistent patterns — some areas use DI, others use globals
7. Scalability bottlenecks — singletons, shared mutable state, synchronous blocking
8. Layering violations — presentation layer knowing about data layer details
9. Missing interfaces/abstractions — hard-coded implementations instead of interfaces
10. Configuration management — hardcoded values, environment concerns

Return a JSON array:
[
  {{
    "title": "Specific architectural issue",
    "severity": "High|Medium|Low|Info",
    "file": "the/file/or/module.py",
    "line": null,
    "description": "Detailed explanation of the architectural problem and its impact",
    "evidence": "Code snippet or structural pattern that demonstrates the issue",
    "recommendation": "Specific architectural improvement with rationale",
    "confidence": 0.80
  }}
]

Return ONLY the JSON array.
"""


def _analyze_structure_metrics(repo_root: str) -> Dict:
    """
    Compute structural metrics deterministically.
    """
    root = Path(repo_root).resolve()

    # Find all Python files and their import counts
    py_files = find_files(repo_root, extension=".py")

    # Count imports per file (coupling indicator)
    from tools.repository import read_file
    import re

    import_counts = {}
    cross_module_imports = {}

    for rel_path in py_files[:30]:
        ok, content = read_file(repo_root, rel_path)
        if not ok:
            continue
        lines = content.splitlines()
        imports = [l for l in lines if l.strip().startswith(("import ", "from "))]
        import_counts[rel_path] = len(imports)

        # Check for circular-like patterns
        local_imports = [
            l for l in imports
            if "from ." in l or (
                "from " in l and not any(
                    lib in l for lib in ["os", "sys", "re", "json", "typing", "pathlib", "abc"]
                )
            )
        ]
        cross_module_imports[rel_path] = local_imports

    # Find largest files (potential god objects)
    file_sizes = {}
    for rel_path in py_files:
        ok, content = read_file(repo_root, rel_path)
        if ok:
            file_sizes[rel_path] = len(content.splitlines())

    large_files = sorted(file_sizes.items(), key=lambda x: x[1], reverse=True)[:5]
    highly_coupled = sorted(import_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "total_modules": len(py_files),
        "large_files": large_files,
        "highly_coupled_files": highly_coupled,
        "avg_imports": sum(import_counts.values()) / max(1, len(import_counts)),
    }


def run_architecture_agent(state: AnalysisState) -> Dict:
    """
    LangGraph node: Architecture Agent.

    Args:
        state: Current graph state.

    Returns:
        State updates dict.
    """
    repo_root = state["repo_path"]
    revision_count = state.get("revision_count", 0)
    logger.info(f"[Architecture] Running (revision {revision_count})")

    reanalysis_files, critic_feedback = get_reanalysis_context(state, "architecture")

    preferred_extensions = [".py", ".js", ".ts", ".java", ".go"]
    files = select_files_for_agent(
        state,
        preferred_extensions=preferred_extensions,
        max_files=12,
        reanalysis_files=reanalysis_files,
    )

    # Get metrics
    try:
        metrics = _analyze_structure_metrics(repo_root)
        metrics_text = (
            f"Total modules: {metrics['total_modules']}\n"
            f"Average imports per module: {metrics['avg_imports']:.1f}\n"
            f"Largest files (lines):\n" +
            "\n".join(f"  {f}: {lines} lines" for f, lines in metrics["large_files"]) +
            f"\nHighly coupled modules (by import count):\n" +
            "\n".join(f"  {f}: {count} imports" for f, count in metrics["highly_coupled_files"])
        )
    except Exception:
        metrics_text = "Metrics unavailable"

    # Add deterministic large file findings
    det_findings: List[Finding] = []
    try:
        if metrics and metrics["large_files"]:
            for file_path, line_count in metrics["large_files"][:3]:
                if line_count > 500:
                    det_findings.append(Finding(
                        category=FindingCategory.ARCHITECTURE,
                        title=f"Large module: {Path(file_path).name} ({line_count} lines)",
                        severity=Severity.MEDIUM,
                        file=file_path,
                        line=None,
                        description=(
                            f"This file is {line_count} lines long. "
                            "Very large files often indicate a God object or module "
                            "with too many responsibilities, making it hard to maintain, "
                            "test, and reason about."
                        ),
                        evidence=f"File size: {line_count} lines",
                        recommendation=(
                            "Apply the Single Responsibility Principle. "
                            "Break the module into smaller, focused modules. "
                            "Consider using dependency injection to decouple components."
                        ),
                        confidence=0.90,
                        source=FindingSource.DETERMINISTIC,
                    ))
    except Exception:
        pass

    repo_summary = state.get("repo_summary")
    languages = ", ".join(repo_summary.languages) if repo_summary else "Unknown"
    frameworks = ", ".join(repo_summary.frameworks) if repo_summary else "Unknown"
    plan = state.get("investigation_plan")
    focus_areas = ", ".join(plan.focus_areas) if plan else "architecture"

    tree = repository_tree(repo_root)
    file_contents = read_files_for_analysis(repo_root, files[:8])
    code_context = build_code_context(file_contents, max_total_chars=10000)

    prompt = ARCHITECTURE_PROMPT.format(
        critic_feedback=critic_feedback or "",
        languages=languages,
        frameworks=frameworks,
        focus_areas=focus_areas,
        directory_tree=tree[:3000],
        code_context=code_context,
        metrics=metrics_text,
    )

    llm_findings = []
    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        llm_findings = parse_findings_from_llm(
            response.content,
            FindingCategory.ARCHITECTURE,
            "Architecture",
        )
    except Exception as exc:
        logger.error(f"[Architecture] LLM error: {exc}", exc_info=True)

    all_findings = det_findings + llm_findings
    summary = f"Architecture review of {len(files)} files: {len(all_findings)} issues found."

    trace_entry: AgentTraceEntry = {
        "node": "Architecture Agent",
        "iteration": revision_count + 1,
        "action": f"Analyzed {len(files)} files and directory structure",
        "result": f"Found {len(all_findings)} architectural issues",
    }

    logger.info(f"[Architecture] Complete: {len(all_findings)} findings")

    return {
        "all_findings": all_findings,
        "architecture_summary": summary,
        "architecture_files": files,
        "agent_trace": [trace_entry],
    }
