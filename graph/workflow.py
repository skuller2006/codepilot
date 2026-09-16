"""
LangGraph Workflow — CodePilot AI.

Defines the StateGraph that orchestrates all agents.

Workflow:
  repository_analyzer → planner
                          ↓
          code_review ─── security ─── testing (parallel fan-out)
                          ↓
                      architecture
                          ↓
                      documentation
                          ↓
                  findings_aggregator
                          ↓
                        critic
                       ↙      ↘
                reporter   re_investigate
                               ↓
                  (appropriate agent re-runs)
                               ↓
                    findings_aggregator
                               ↓
                            critic
                           (max 3 cycles)

Conditional edges:
  - After critic: route to reporter or re_investigate
  - After re_investigate: route specific agents based on critic feedback
"""

from __future__ import annotations

import logging
from typing import List, Literal

from langgraph.graph import END, StateGraph

from agents.aggregator import run_findings_aggregator
from agents.architecture import run_architecture_agent
from agents.code_review import run_code_review
from agents.critic import critic_routing, run_critic
from agents.documentation import run_documentation_agent
from agents.planner import run_planner
from agents.reporter import run_reporter
from agents.repository_analyzer import run_repository_analyzer
from agents.security import run_security_agent
from agents.testing import run_testing_agent
from config import MAX_REVISION_CYCLES
from graph.state import AnalysisState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Re-Investigation Node
# ---------------------------------------------------------------------------

def run_re_investigate(state: AnalysisState) -> dict:
    """
    Increments the revision counter and runs the agents that the
    critic flagged for re-analysis.

    This node is visited only when the Critic rejects findings.
    The new findings are returned as a list; LangGraph's _merge_findings
    reducer will append them to the existing all_findings.
    """
    revision_count = state.get("revision_count", 0) + 1
    logger.info(f"[ReInvestigate] Starting revision cycle {revision_count}")

    # Determine which agents to re-run
    requests = state.get("reanalysis_requests", [])
    agents_to_rerun = {r.agent_name for r in requests}

    if not agents_to_rerun:
        # Fallback: re-run security and code review
        agents_to_rerun = {"security", "code_review"}

    # Build updated state for passing into sub-agents
    sub_state = {**state, "revision_count": revision_count}

    new_findings = []
    new_traces = []
    summary_updates: dict = {"revision_count": revision_count}

    AGENT_MAP = {
        "security": run_security_agent,
        "code_review": run_code_review,
        "testing": run_testing_agent,
        "architecture": run_architecture_agent,
        "documentation": run_documentation_agent,
    }

    for agent_name in agents_to_rerun:
        agent_fn = AGENT_MAP.get(agent_name)
        if not agent_fn:
            logger.warning(f"[ReInvestigate] Unknown agent: {agent_name}")
            continue

        logger.info(f"[ReInvestigate] Re-running {agent_name}")
        try:
            result = agent_fn(sub_state)
            # Collect new findings
            new_findings.extend(result.get("all_findings", []))
            new_traces.extend(result.get("agent_trace", []))
            # Collect summary updates (not findings/trace)
            for k, v in result.items():
                if k not in ("all_findings", "agent_trace"):
                    summary_updates[k] = v
        except Exception as exc:
            logger.error(f"[ReInvestigate] Agent {agent_name} failed: {exc}", exc_info=True)
            summary_updates.setdefault("errors", []).append(
                f"Re-investigation of {agent_name} failed: {str(exc)}"
            )

    # Return new findings so reducer can merge them
    summary_updates["all_findings"] = new_findings
    summary_updates["agent_trace"] = new_traces
    return summary_updates


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def build_workflow() -> StateGraph:
    """
    Build and compile the CodePilot AI LangGraph workflow.

    Returns:
        Compiled StateGraph ready to invoke.
    """
    workflow = StateGraph(AnalysisState)

    # Add all nodes
    workflow.add_node("repository_analyzer", run_repository_analyzer)
    workflow.add_node("planner", run_planner)
    workflow.add_node("code_review", run_code_review)
    workflow.add_node("security", run_security_agent)
    workflow.add_node("testing", run_testing_agent)
    workflow.add_node("architecture", run_architecture_agent)
    workflow.add_node("documentation", run_documentation_agent)
    workflow.add_node("findings_aggregator", run_findings_aggregator)
    workflow.add_node("critic", run_critic)
    workflow.add_node("re_investigate", run_re_investigate)
    workflow.add_node("reporter", run_reporter)

    # Entry point
    workflow.set_entry_point("repository_analyzer")

    # Linear edges: repo → plan
    workflow.add_edge("repository_analyzer", "planner")

    # Fan-out: planner → all three parallel agents
    workflow.add_edge("planner", "code_review")
    workflow.add_edge("planner", "security")
    workflow.add_edge("planner", "testing")

    # Fan-in: all three → architecture (sequential)
    workflow.add_edge("code_review", "architecture")
    workflow.add_edge("security", "architecture")
    workflow.add_edge("testing", "architecture")

    # Architecture → documentation → aggregator
    workflow.add_edge("architecture", "documentation")
    workflow.add_edge("documentation", "findings_aggregator")

    # Aggregator → critic
    workflow.add_edge("findings_aggregator", "critic")

    # Conditional edge: critic → reporter OR re_investigate
    workflow.add_conditional_edges(
        "critic",
        critic_routing,
        {
            "reporter": "reporter",
            "re_investigate": "re_investigate",
        },
    )

    # Re-investigation → aggregator (re-aggregate, then critic again)
    workflow.add_edge("re_investigate", "findings_aggregator")

    # Reporter → END
    workflow.add_edge("reporter", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Workflow Execution
# ---------------------------------------------------------------------------

def run_analysis(
    repo_path: str,
    user_request: str = "",
    focus_areas: List[str] = None,
    progress_callback=None,
) -> AnalysisState:
    """
    Run the complete CodePilot AI analysis workflow.

    Args:
        repo_path: Absolute path to the extracted repository.
        user_request: User's natural language request/focus.
        focus_areas: List of focus area strings.
        progress_callback: Optional callback(node_name, status) for UI updates.

    Returns:
        Final AnalysisState after workflow completion.
    """
    app = build_workflow()

    initial_state: AnalysisState = {
        "repo_path": repo_path,
        "user_request": user_request,
        "focus_areas": focus_areas or [],
        "repo_summary": None,
        "investigation_plan": None,
        "all_findings": [],
        "code_review_summary": "",
        "security_summary": "",
        "testing_summary": "",
        "architecture_summary": "",
        "documentation_summary": "",
        "code_review_files": [],
        "security_files": [],
        "testing_files": [],
        "architecture_files": [],
        "documentation_files": [],
        "critic_review": None,
        "revision_count": 0,
        "reanalysis_requests": [],
        "final_report": None,
        "agent_trace": [],
        "current_node": "repository_analyzer",
        "errors": [],
    }

    config = {"recursion_limit": 50}

    logger.info(f"[Workflow] Starting analysis of {repo_path}")

    try:
        final_state = app.invoke(initial_state, config=config)
        logger.info(f"[Workflow] Complete: {len(final_state.get('all_findings', []))} findings")
        return final_state
    except Exception as exc:
        logger.error(f"[Workflow] Fatal error: {exc}", exc_info=True)
        initial_state["errors"] = [f"Workflow error: {str(exc)}"]
        return initial_state
