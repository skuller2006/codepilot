"""
Findings Aggregator — CodePilot AI.

Collects all agent findings, deduplicates them, and prepares
the consolidated set for Critic review.

This is a lightweight aggregation step — not an LLM call.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from graph.state import AnalysisState, AgentTraceEntry
from models.schemas import Finding, FindingCategory, Severity

logger = logging.getLogger(__name__)


def _deduplicate_findings(findings: List[Finding]) -> List[Finding]:
    """
    Remove near-duplicate findings based on title and file similarity.
    """
    seen_keys = set()
    unique = []
    for f in findings:
        # Create a dedup key: normalized title + file
        title_key = f.title.lower().strip()[:50]
        file_key = (f.file or "").lower()
        key = f"{title_key}|{file_key}"
        if key not in seen_keys:
            seen_keys.add(key)
            unique.append(f)
    return unique


def _sort_by_severity(findings: List[Finding]) -> List[Finding]:
    """Sort findings by severity (Critical first)."""
    severity_order = {
        Severity.CRITICAL: 0,
        Severity.HIGH: 1,
        Severity.MEDIUM: 2,
        Severity.LOW: 3,
        Severity.INFO: 4,
    }
    return sorted(findings, key=lambda f: severity_order.get(f.severity, 5))


def run_findings_aggregator(state: AnalysisState) -> Dict:
    """
    LangGraph node: Findings Aggregator.

    Deduplicates and prioritizes all collected findings.

    Args:
        state: Current graph state.

    Returns:
        State updates with deduplicated, sorted findings.
    """
    all_findings = state.get("all_findings", [])
    logger.info(f"[Aggregator] Aggregating {len(all_findings)} findings")

    # Deduplicate
    deduped = _deduplicate_findings(all_findings)

    # Sort by severity
    sorted_findings = _sort_by_severity(deduped)

    # Summary stats
    by_category: Dict[str, int] = {}
    by_severity: Dict[str, int] = {}
    for f in sorted_findings:
        by_category[f.category.value] = by_category.get(f.category.value, 0) + 1
        by_severity[f.severity.value] = by_severity.get(f.severity.value, 0) + 1

    removed = len(all_findings) - len(deduped)
    summary_parts = [
        f"Total: {len(sorted_findings)} findings",
        f"Duplicates removed: {removed}",
        "By severity: " + ", ".join(f"{k}: {v}" for k, v in by_severity.items()),
        "By category: " + ", ".join(f"{k}: {v}" for k, v in by_category.items()),
    ]

    trace_entry: AgentTraceEntry = {
        "node": "Findings Aggregator",
        "iteration": state.get("revision_count", 0) + 1,
        "action": f"Aggregated and deduplicated {len(all_findings)} findings",
        "result": f"{len(sorted_findings)} unique findings: " + ", ".join(
            f"{v} {k}" for k, v in by_severity.items()
        ),
    }

    logger.info(f"[Aggregator] {len(sorted_findings)} unique findings after dedup")

    return {
        "all_findings": sorted_findings,
        "agent_trace": [trace_entry],
        "current_node": "critic",
    }
