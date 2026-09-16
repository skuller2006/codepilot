"""
Code Review Agent — CodePilot AI.

Analyzes source files for bugs, code smells, error handling issues,
maintainability problems, and performance concerns.
"""

from __future__ import annotations

import logging
from typing import Dict

from config import get_llm
from graph.state import AnalysisState, AgentTraceEntry
from models.schemas import FindingCategory, FindingSource, Severity
from agents.base import (
    build_code_context,
    get_reanalysis_context,
    parse_findings_from_llm,
    read_files_for_analysis,
    select_files_for_agent,
)
from tools.code_search import search_code

logger = logging.getLogger(__name__)

CODE_REVIEW_PROMPT = """You are a senior software engineer performing a thorough code review.

Analyze the following source files and identify concrete issues.

{critic_feedback}

Repository context:
- Languages: {languages}
- Frameworks: {frameworks}
- Investigation focus: {focus_areas}

Source files:
{code_context}

Search results for common anti-patterns:
{search_results}

Find REAL issues only — do not invent problems. Each finding must have code evidence.

Return a JSON array of findings:
[
  {{
    "title": "Concise title of the issue",
    "severity": "Critical|High|Medium|Low|Info",
    "file": "relative/path/to/file.py",
    "line": 42,
    "description": "Detailed explanation of why this is a problem",
    "evidence": "The actual code snippet that demonstrates the issue",
    "recommendation": "Specific, actionable fix",
    "confidence": 0.85
  }}
]

Focus on:
1. Bugs and logic errors
2. Poor error handling (bare excepts, swallowed exceptions)
3. Code smells (long functions, deep nesting, magic numbers)
4. Resource leaks (unclosed files, connections)
5. Thread safety issues
6. Performance anti-patterns (N+1 queries, unnecessary loops)
7. Duplicated logic
8. Maintainability issues

Return ONLY the JSON array. If no significant issues found, return [].
"""


def run_code_review(state: AnalysisState) -> Dict:
    """
    LangGraph node: Code Review Agent.

    Args:
        state: Current graph state.

    Returns:
        State updates dict.
    """
    repo_root = state["repo_path"]
    revision_count = state.get("revision_count", 0)
    logger.info(f"[CodeReview] Running (revision {revision_count})")

    reanalysis_files, critic_feedback = get_reanalysis_context(state, "code_review")

    preferred_extensions = [".py", ".js", ".ts", ".java", ".go", ".rb", ".php", ".cs"]
    files = select_files_for_agent(
        state,
        preferred_extensions=preferred_extensions,
        reanalysis_files=reanalysis_files,
    )

    if not files:
        return {
            "agent_trace": [{
                "node": "Code Review Agent",
                "iteration": revision_count + 1,
                "action": "No source files found to analyze",
                "result": "Skipped",
            }],
            "code_review_summary": "No source files found.",
            "code_review_files": [],
        }

    file_contents = read_files_for_analysis(repo_root, files)
    code_context = build_code_context(file_contents)

    # Deterministic searches to augment LLM analysis
    search_results_parts = []
    anti_patterns = [
        ("bare except", r"except\s*:", True),
        ("eval/exec usage", r"\beval\s*\(|\bexec\s*\(", True),
        ("TODO/FIXME markers", r"(?i)\b(TODO|FIXME|HACK|XXX)\b", False),
        ("print debugging", r"\bprint\s*\(", False),
    ]
    for label, pattern, is_regex in anti_patterns:
        results = search_code(repo_root, pattern, is_regex=is_regex, max_results=5)
        if results and not results[0].get("error"):
            search_results_parts.append(
                f"\n{label}:\n" + "\n".join(
                    f"  {r['file']}:{r['line_number']}: {r['line_content'][:100]}"
                    for r in results
                )
            )
    search_results_text = "".join(search_results_parts) or "No patterns found."

    repo_summary = state.get("repo_summary")
    languages = ", ".join(repo_summary.languages) if repo_summary else "Unknown"
    frameworks = ", ".join(repo_summary.frameworks) if repo_summary else "Unknown"
    plan = state.get("investigation_plan")
    focus_areas = ", ".join(plan.focus_areas) if plan else "general code quality"

    prompt = CODE_REVIEW_PROMPT.format(
        critic_feedback=critic_feedback or "",
        languages=languages,
        frameworks=frameworks,
        focus_areas=focus_areas,
        code_context=code_context,
        search_results=search_results_text,
    )

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        findings = parse_findings_from_llm(
            response.content,
            FindingCategory.CODE_QUALITY,
            "CodeReview",
        )

        # Also add deterministic findings for obvious patterns
        for result in search_code(repo_root, r"except\s*:", is_regex=True, max_results=3):
            if not result.get("error"):
                from models.schemas import Finding
                findings.append(Finding(
                    category=FindingCategory.BUG,
                    title="Bare except clause — swallows all exceptions",
                    severity=Severity.MEDIUM,
                    file=result["file"],
                    line=result["line_number"],
                    description="A bare `except:` clause catches all exceptions including SystemExit and KeyboardInterrupt, masking real errors.",
                    evidence=result["line_content"],
                    recommendation="Catch specific exception types (e.g., `except ValueError:`) and log or handle them explicitly.",
                    confidence=0.95,
                    source=FindingSource.DETERMINISTIC,
                ))

        summary = f"Code review analyzed {len(files)} files, found {len(findings)} issues."

        trace_entry: AgentTraceEntry = {
            "node": "Code Review Agent",
            "iteration": revision_count + 1,
            "action": f"Analyzed {len(files)} source files",
            "result": f"Found {len(findings)} findings",
        }

        logger.info(f"[CodeReview] Complete: {len(findings)} findings in {len(files)} files")

        return {
            "all_findings": findings,
            "code_review_summary": summary,
            "code_review_files": files,
            "agent_trace": [trace_entry],
        }

    except Exception as exc:
        logger.error(f"[CodeReview] Error: {exc}", exc_info=True)
        return {
            "errors": [f"Code Review Agent error: {str(exc)}"],
            "code_review_summary": f"Error during analysis: {str(exc)}",
            "code_review_files": files,
            "agent_trace": [{
                "node": "Code Review Agent",
                "iteration": revision_count + 1,
                "action": "Analysis failed",
                "result": str(exc)[:200],
            }],
        }
