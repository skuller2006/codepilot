"""
Testing Agent — CodePilot AI.

Analyzes test coverage, identifies untested modules, missing edge cases,
and recommends specific tests to add.
"""

from __future__ import annotations

import logging
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
from tools.repository import find_files

logger = logging.getLogger(__name__)

TESTING_PROMPT = """You are a senior QA engineer and testing expert analyzing a codebase.

{critic_feedback}

Repository context:
- Languages: {languages}
- Frameworks: {frameworks}
- Test files found: {test_file_count}
- Source files: {source_file_count}
- Focus: {focus_areas}

Source files (modules to be tested):
{source_context}

Existing test files:
{test_context}

Testing statistics:
{test_stats}

Analyze and identify:
1. Which source modules have NO tests
2. Test files that exist but have poor coverage (too few test cases for the complexity)
3. Missing edge case tests (boundary conditions, empty inputs, large inputs)
4. Missing error path tests (exception handling, invalid inputs, network failures)
5. Missing integration tests (if complex interactions exist between modules)
6. Test quality issues (tests that don't actually assert anything useful)
7. Missing test fixtures or setup

Return a JSON array:
[
  {{
    "title": "Missing tests for X module/function",
    "severity": "High|Medium|Low|Info",
    "file": "path/to/source/file.py",
    "line": null,
    "description": "Detailed explanation of what's missing and why it matters",
    "evidence": "The untested function/class code or the relevant code snippet",
    "recommendation": "Specific test cases to add, with example test code if helpful",
    "confidence": 0.85
  }}
]

Be specific. Name exact functions, classes, and scenarios that need testing.
Return ONLY the JSON array.
"""


def _analyze_test_coverage_deterministically(repo_root: str) -> Dict:
    """
    Analyze test structure deterministically without LLM.
    """
    # Find test files
    test_files = []
    for ext in [".py", ".js", ".ts", ".java", ".go", ".rb"]:
        ext_files = find_files(repo_root, extension=ext)
        test_files.extend([
            f for f in ext_files
            if "test" in f.lower() or "spec" in f.lower()
        ])

    # Find source files (non-test)
    source_files = []
    for ext in [".py", ".js", ".ts", ".java", ".go", ".rb"]:
        ext_files = find_files(repo_root, extension=ext)
        source_files.extend([
            f for f in ext_files
            if "test" not in f.lower() and "spec" not in f.lower()
        ])

    # Find test config files
    test_configs = []
    for cfg in ["pytest.ini", "setup.cfg", "jest.config.js", "karma.conf.js", "phpunit.xml", "tox.ini", ".coveragerc"]:
        matches = find_files(repo_root, name_pattern=cfg)
        test_configs.extend(matches)

    # Estimate module coverage (which source dirs have corresponding test dirs)
    from pathlib import Path
    source_modules = set(Path(f).parent for f in source_files)
    tested_modules = set()
    for tf in test_files:
        # Look for corresponding source files
        tf_path = Path(tf)
        name_without_test = tf_path.stem.replace("test_", "").replace("_test", "").replace(".test", "").replace(".spec", "")
        for sf in source_files:
            if name_without_test in Path(sf).stem:
                tested_modules.add(Path(sf).parent)
                break

    untested_source_files = [
        f for f in source_files
        if not any(
            Path(f).stem in tf or Path(f).stem.replace("-", "_") in tf
            for tf in test_files
        )
    ]

    return {
        "test_files": test_files[:30],
        "source_files": source_files[:50],
        "test_config_files": test_configs,
        "test_count": len(test_files),
        "source_count": len(source_files),
        "untested_source_files": untested_source_files[:20],
        "estimated_coverage": (
            f"~{min(100, int((len(source_files) - len(untested_source_files)) / max(1, len(source_files)) * 100))}%"
            if source_files else "N/A"
        ),
    }


def run_testing_agent(state: AnalysisState) -> Dict:
    """
    LangGraph node: Testing Agent.

    Args:
        state: Current graph state.

    Returns:
        State updates dict.
    """
    repo_root = state["repo_path"]
    revision_count = state.get("revision_count", 0)
    logger.info(f"[Testing] Running (revision {revision_count})")

    reanalysis_files, critic_feedback = get_reanalysis_context(state, "testing")

    # Deterministic analysis
    test_stats = _analyze_test_coverage_deterministically(repo_root)

    # Select source files to analyze for missing tests
    preferred_extensions = [".py", ".js", ".ts", ".java", ".go"]
    source_files_to_analyze = [
        f for f in select_files_for_agent(
            state,
            preferred_extensions=preferred_extensions,
            max_files=10,
            reanalysis_files=reanalysis_files,
        )
        if "test" not in f.lower() and "spec" not in f.lower()
    ]

    # Read source and test files
    source_contents = read_files_for_analysis(repo_root, source_files_to_analyze[:8])
    test_contents = read_files_for_analysis(repo_root, test_stats["test_files"][:5])

    source_context = build_code_context(source_contents, max_total_chars=12000)
    test_context = build_code_context(test_contents, max_total_chars=8000)

    stats_text = (
        f"Test files: {test_stats['test_count']}\n"
        f"Source files: {test_stats['source_count']}\n"
        f"Estimated test coverage: {test_stats['estimated_coverage']}\n"
        f"Test configs: {', '.join(test_stats['test_config_files']) or 'None'}\n"
        f"Potentially untested files:\n" +
        "\n".join(f"  - {f}" for f in test_stats["untested_source_files"][:10])
    )

    repo_summary = state.get("repo_summary")
    languages = ", ".join(repo_summary.languages) if repo_summary else "Unknown"
    frameworks = ", ".join(repo_summary.frameworks) if repo_summary else "Unknown"
    plan = state.get("investigation_plan")
    focus_areas = ", ".join(plan.focus_areas) if plan else "testing"

    prompt = TESTING_PROMPT.format(
        critic_feedback=critic_feedback or "",
        languages=languages,
        frameworks=frameworks,
        test_file_count=test_stats["test_count"],
        source_file_count=test_stats["source_count"],
        focus_areas=focus_areas,
        source_context=source_context or "No source files available.",
        test_context=test_context or "No test files found.",
        test_stats=stats_text,
    )

    llm_findings = []
    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        llm_findings = parse_findings_from_llm(
            response.content,
            FindingCategory.TESTING,
            "Testing",
        )
    except Exception as exc:
        logger.error(f"[Testing] LLM error: {exc}", exc_info=True)

    # Add deterministic finding for missing tests entirely
    det_findings: List[Finding] = []
    if test_stats["test_count"] == 0 and test_stats["source_count"] > 0:
        det_findings.append(Finding(
            category=FindingCategory.TESTING,
            title="No test files found in repository",
            severity=Severity.HIGH,
            file=None,
            line=None,
            description=(
                f"The repository contains {test_stats['source_count']} source files "
                "but has no test files. This represents significant risk — "
                "bugs can be introduced without automatic detection."
            ),
            evidence=f"Source files: {test_stats['source_count']}, Test files: 0",
            recommendation=(
                "Add a testing framework (pytest for Python, Jest for JS) "
                "and create unit tests for all critical paths and business logic."
            ),
            confidence=1.0,
            source=FindingSource.DETERMINISTIC,
        ))

    all_findings = det_findings + llm_findings
    summary = (
        f"Testing analysis: {test_stats['test_count']} test files, "
        f"{test_stats['source_count']} source files, "
        f"estimated coverage {test_stats['estimated_coverage']}. "
        f"Found {len(all_findings)} testing gaps."
    )

    trace_entry: AgentTraceEntry = {
        "node": "Testing Agent",
        "iteration": revision_count + 1,
        "action": f"Analyzed {test_stats['source_count']} source files and {test_stats['test_count']} test files",
        "result": f"Found {len(all_findings)} testing gaps, estimated coverage: {test_stats['estimated_coverage']}",
    }

    logger.info(f"[Testing] Complete: {len(all_findings)} findings")

    return {
        "all_findings": all_findings,
        "testing_summary": summary,
        "testing_files": source_files_to_analyze + test_stats["test_files"][:5],
        "agent_trace": [trace_entry],
    }
