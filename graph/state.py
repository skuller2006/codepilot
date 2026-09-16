"""
LangGraph shared state definition for CodePilot AI.

All nodes in the graph read from and write to this shared state.
Using TypedDict for LangGraph compatibility with Annotated reducers.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
import operator

from models.schemas import (
    CriticReview,
    Finding,
    FinalReport,
    InvestigationPlan,
    ReanalysisRequest,
    RepositorySummary,
)


def _merge_findings(existing: List[Finding], new: List[Finding]) -> List[Finding]:
    """Reducer that appends new findings, avoiding exact duplicates by title."""
    existing_titles = {f.title for f in existing}
    deduplicated = [f for f in new if f.title not in existing_titles]
    return existing + deduplicated


class AgentTraceEntry(TypedDict):
    """One entry in the execution trace shown in the UI."""

    node: str
    iteration: int
    action: str
    result: str


class AnalysisState(TypedDict):
    """
    Shared state flowing through the entire LangGraph workflow.

    Each agent reads relevant fields and writes its outputs.
    Reducers handle list merging so parallel agents don't overwrite each other.
    """

    # --- Input ---
    repo_path: str                          # Extracted repository path on disk
    user_request: str                        # User's focus instructions
    focus_areas: List[str]                   # Parsed focus areas from user request

    # --- Repository Analysis ---
    repo_summary: Optional[RepositorySummary]

    # --- Planning ---
    investigation_plan: Optional[InvestigationPlan]

    # --- Agent Findings (merged by reducer) ---
    all_findings: Annotated[List[Finding], _merge_findings]

    # --- Per-agent summaries ---
    code_review_summary: str
    security_summary: str
    testing_summary: str
    architecture_summary: str
    documentation_summary: str

    # --- Files analyzed per agent ---
    code_review_files: List[str]
    security_files: List[str]
    testing_files: List[str]
    architecture_files: List[str]
    documentation_files: List[str]

    # --- Critic / Revision loop ---
    critic_review: Optional[CriticReview]
    revision_count: int                      # How many critic→re-investigate cycles
    reanalysis_requests: List[ReanalysisRequest]

    # --- Final output ---
    final_report: Optional[FinalReport]

    # --- UI / Tracing ---
    agent_trace: Annotated[List[AgentTraceEntry], operator.add]
    current_node: str
    errors: Annotated[List[str], operator.add]
