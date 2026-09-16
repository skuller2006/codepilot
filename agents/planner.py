"""
Planning Agent — CodePilot AI.

Creates an investigation plan based on the repository summary and user request.
Uses structured LLM output to determine which agents and areas to focus on.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List

from config import get_llm
from graph.state import AnalysisState, AgentTraceEntry
from models.schemas import InvestigationPlan

logger = logging.getLogger(__name__)

KNOWN_AGENTS = [
    "code_review",
    "security",
    "testing",
    "architecture",
    "documentation",
]


def _build_planning_prompt(state: AnalysisState) -> str:
    """Build the planning prompt from state."""
    repo_summary = state.get("repo_summary")
    user_request = state.get("user_request", "")
    focus_areas = state.get("focus_areas", [])

    if repo_summary is None:
        repo_info = "Repository information unavailable."
    else:
        repo_info = f"""
Languages: {', '.join(repo_summary.languages)}
Frameworks: {', '.join(repo_summary.frameworks) or 'None detected'}
Total files: {repo_summary.total_files}
Test files: {repo_summary.test_files}
Has README: {repo_summary.has_readme}
Has Tests: {repo_summary.has_tests}
Has CI: {repo_summary.has_ci}
Dependencies: {repo_summary.dependencies.get('total', 0)} packages
Key structural findings:
{chr(10).join(f'- {f}' for f in repo_summary.key_findings)}

Directory tree (excerpt):
{repo_summary.directory_tree[:2000]}
"""

    focus_text = f"User focus areas: {', '.join(focus_areas)}" if focus_areas else ""
    user_request_text = f"User request: {user_request}" if user_request else ""

    return f"""You are a senior software engineering team lead planning a code review.

Repository Information:
{repo_info}

{user_request_text}
{focus_text}

Available specialist agents: {', '.join(KNOWN_AGENTS)}

Create an investigation plan as a JSON object with exactly these fields:
{{
  "focus_areas": ["list of specific technical areas to investigate, e.g. authentication, database queries, error handling"],
  "agents_required": ["list of agent names from: {', '.join(KNOWN_AGENTS)}"],
  "rationale": "1-2 sentence explanation of why this plan was chosen",
  "key_files_to_examine": ["list of specific files that agents should examine closely, based on the tree"]
}}

Rules:
- Always include at least 3 agents
- focus_areas should be specific (e.g. "JWT authentication" not just "authentication")
- key_files_to_examine should be real file paths from the tree, max 15 files
- If user specified focus areas, weight those heavily

Return ONLY the JSON object, no other text.
"""


def run_planner(state: AnalysisState) -> Dict:
    """
    LangGraph node: Planning Agent.

    Determines which agents to run and what to focus on.

    Args:
        state: Current graph state.

    Returns:
        State updates dict.
    """
    logger.info("[Planner] Creating investigation plan")

    try:
        llm = get_llm()
        prompt = _build_planning_prompt(state)
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Extract JSON from response (handle markdown code blocks)
        if "```" in content:
            import re
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if match:
                content = match.group(1)

        plan_data = json.loads(content)

        # Validate and normalize agents_required
        valid_agents = [
            a for a in plan_data.get("agents_required", KNOWN_AGENTS)
            if a in KNOWN_AGENTS
        ]
        if not valid_agents:
            valid_agents = KNOWN_AGENTS

        plan = InvestigationPlan(
            focus_areas=plan_data.get("focus_areas", ["general code quality"]),
            agents_required=valid_agents,
            rationale=plan_data.get("rationale", "Standard comprehensive review"),
            key_files_to_examine=plan_data.get("key_files_to_examine", [])[:15],
        )

        trace_entry: AgentTraceEntry = {
            "node": "Planning Agent",
            "iteration": 1,
            "action": f"Created investigation plan with {len(valid_agents)} agents",
            "result": f"Focus areas: {', '.join(plan.focus_areas[:3])}{'...' if len(plan.focus_areas) > 3 else ''}",
        }

        logger.info(f"[Planner] Plan: agents={valid_agents}, areas={plan.focus_areas[:3]}")

        return {
            "investigation_plan": plan,
            "agent_trace": [trace_entry],
            "current_node": "specialist_agents",
        }

    except json.JSONDecodeError as exc:
        logger.warning(f"[Planner] JSON parse error: {exc}, using default plan")
        # Fall back to running all agents
        plan = InvestigationPlan(
            focus_areas=state.get("focus_areas", ["security", "code quality", "testing"]),
            agents_required=KNOWN_AGENTS,
            rationale="Default plan used due to planning agent JSON parse error.",
            key_files_to_examine=[],
        )
        return {
            "investigation_plan": plan,
            "agent_trace": [{
                "node": "Planning Agent",
                "iteration": 1,
                "action": "Used default plan (JSON parse error)",
                "result": f"All {len(KNOWN_AGENTS)} agents will run",
            }],
            "current_node": "specialist_agents",
        }

    except Exception as exc:
        logger.error(f"[Planner] Error: {exc}", exc_info=True)
        plan = InvestigationPlan(
            focus_areas=["security", "code quality", "testing"],
            agents_required=KNOWN_AGENTS,
            rationale=f"Default plan (planner error: {str(exc)[:100]})",
            key_files_to_examine=[],
        )
        return {
            "investigation_plan": plan,
            "errors": [f"Planner error: {str(exc)}"],
            "agent_trace": [{
                "node": "Planning Agent",
                "iteration": 1,
                "action": "Error — using default plan",
                "result": str(exc)[:200],
            }],
            "current_node": "specialist_agents",
        }
