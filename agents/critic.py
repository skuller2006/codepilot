"""
Critic Agent — CodePilot AI.

The Critic is a core component of the LangGraph workflow.
It reviews all findings and either:
  - Approves them → workflow proceeds to report generation
  - Rejects them → workflow loops back to the appropriate specialist agents

This feedback loop is what makes the system genuinely agentic.
The Critic checks:
  1. Sufficient code evidence for each finding
  2. Finding validity (not a false positive)
  3. Severity accuracy
  4. Recommendation quality
  5. Duplicate detection
  6. Missing investigation areas
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List

from config import MAX_REVISION_CYCLES, get_llm
from graph.state import AnalysisState, AgentTraceEntry
from models.schemas import CriticReview, Finding, ReanalysisRequest, Severity

logger = logging.getLogger(__name__)

CRITIC_PROMPT = """You are a critical senior engineer reviewing AI-generated code analysis findings.

Your job is to quality-check these findings and decide if they are ready for a final report.

Repository context:
- Languages: {languages}
- Frameworks: {frameworks}

Current revision cycle: {revision_count} of {max_revisions}

All findings to review:
{findings_text}

Evaluate EACH finding for:
1. CODE EVIDENCE — Does the finding cite specific code? Is the evidence convincing?
2. VALIDITY — Is this actually a real problem, or a false positive?
3. SEVERITY — Is the severity appropriate for the actual risk?
4. RECOMMENDATION — Is the recommendation specific and actionable?
5. DUPLICATES — Are any findings essentially the same issue?

Also consider:
6. Are there important areas that were NOT investigated?
7. Are critical/high severity findings well-supported?

Return a JSON object (no other text):
{{
  "approved": true or false,
  "overall_quality": 0-100,
  "invalid_findings": ["title of finding 1 that lacks evidence", "..."],
  "missing_investigations": ["area that should have been investigated", "..."],
  "required_reanalysis": [
    {{
      "agent_name": "security",
      "reason": "Why re-investigation is needed",
      "focus_areas": ["specific area 1", "specific area 2"],
      "specific_files": ["file1.py", "file2.py"]
    }}
  ],
  "duplicate_findings": ["title of duplicate 1", "..."],
  "comments": ["Specific feedback comment 1", "..."]
}}

Rules:
- Set "approved": true if overall_quality >= 70 AND no critical findings are unsupported
- Set "approved": false if ANY critical/high finding lacks code evidence
- required_reanalysis must reference valid agents: security, code_review, testing, architecture, documentation
- Be strict but fair — don't reject findings that are valid just because they're not perfect
- If revision_count >= {max_revisions}, approve regardless (max iterations reached)

Return ONLY the JSON object.
"""


def _format_findings_for_critic(findings: List[Finding]) -> str:
    """Format findings into a readable text for the critic."""
    parts = []
    for i, f in enumerate(findings, 1):
        evidence = f.evidence[:200] if f.evidence else "NO CODE EVIDENCE PROVIDED"
        parts.append(
            f"FINDING {i}: [{f.severity.value}] {f.title}\n"
            f"  Category: {f.category.value}\n"
            f"  Source: {f.source.value}\n"
            f"  File: {f.file or 'Not specified'}\n"
            f"  Line: {f.line or 'Not specified'}\n"
            f"  Evidence: {evidence}\n"
            f"  Confidence: {f.confidence_pct()}%\n"
            f"  Recommendation: {f.recommendation[:150]}\n"
        )
    return "\n".join(parts) if parts else "No findings to review."


def run_critic(state: AnalysisState) -> Dict:
    """
    LangGraph node: Critic Agent.

    Reviews all findings and decides whether to approve or request re-investigation.

    Args:
        state: Current graph state.

    Returns:
        State updates including CriticReview and routing information.
    """
    findings = state.get("all_findings", [])
    revision_count = state.get("revision_count", 0)
    logger.info(f"[Critic] Reviewing {len(findings)} findings (revision {revision_count})")

    # Max iterations protection — force approve if at limit
    if revision_count >= MAX_REVISION_CYCLES:
        logger.info(f"[Critic] Max revision cycles ({MAX_REVISION_CYCLES}) reached — force approving")
        review = CriticReview(
            approved=True,
            overall_quality=65,
            invalid_findings=[],
            missing_investigations=[],
            required_reanalysis=[],
            duplicate_findings=[],
            comments=[
                f"Maximum revision cycles ({MAX_REVISION_CYCLES}) reached. "
                "Approving findings as-is. Some may require manual verification."
            ],
        )
        trace_entry: AgentTraceEntry = {
            "node": "Critic Agent",
            "iteration": revision_count + 1,
            "action": "Max revision cycles reached — force approving",
            "result": f"Approved {len(findings)} findings (max iterations protection)",
        }
        return {
            "critic_review": review,
            "agent_trace": [trace_entry],
            "current_node": "reporter",
        }

    if not findings:
        # No findings — nothing to approve, go to report
        review = CriticReview(
            approved=True,
            overall_quality=50,
            invalid_findings=[],
            missing_investigations=["No findings were generated — analysis may have failed"],
            required_reanalysis=[],
            duplicate_findings=[],
            comments=["No findings to review. Report will be minimal."],
        )
        return {
            "critic_review": review,
            "agent_trace": [{
                "node": "Critic Agent",
                "iteration": revision_count + 1,
                "action": "No findings to review",
                "result": "Approved (empty findings set)",
            }],
            "current_node": "reporter",
        }

    repo_summary = state.get("repo_summary")
    languages = ", ".join(repo_summary.languages) if repo_summary else "Unknown"
    frameworks = ", ".join(repo_summary.frameworks) if repo_summary else "Unknown"

    findings_text = _format_findings_for_critic(findings)

    prompt = CRITIC_PROMPT.format(
        languages=languages,
        frameworks=frameworks,
        revision_count=revision_count,
        max_revisions=MAX_REVISION_CYCLES,
        findings_text=findings_text[:12000],  # Limit context
    )

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Extract JSON
        if "```" in content:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if match:
                content = match.group(1)

        review_data = json.loads(content)

        # Build reanalysis requests
        reanalysis_requests = []
        for req in review_data.get("required_reanalysis", []):
            try:
                reanalysis_requests.append(ReanalysisRequest(
                    agent_name=req.get("agent_name", ""),
                    reason=req.get("reason", ""),
                    focus_areas=req.get("focus_areas", []),
                    specific_files=req.get("specific_files", []),
                ))
            except Exception:
                pass

        # Override approval if max iterations reached
        approved = review_data.get("approved", False)

        review = CriticReview(
            approved=approved,
            overall_quality=int(review_data.get("overall_quality", 70)),
            invalid_findings=review_data.get("invalid_findings", []),
            missing_investigations=review_data.get("missing_investigations", []),
            required_reanalysis=[r.agent_name for r in reanalysis_requests],
            duplicate_findings=review_data.get("duplicate_findings", []),
            comments=review_data.get("comments", []),
        )

        if approved:
            action_desc = f"Approved {len(findings)} findings (quality: {review.overall_quality}/100)"
            next_node = "reporter"
        else:
            agents_needed = [r.agent_name for r in reanalysis_requests]
            action_desc = f"Rejected — requested re-analysis from: {', '.join(agents_needed)}"
            next_node = "re_investigate"

        trace_entry = {
            "node": "Critic Agent",
            "iteration": revision_count + 1,
            "action": action_desc,
            "result": (
                f"Quality score: {review.overall_quality}/100. "
                f"Invalid: {len(review.invalid_findings)}. "
                f"Missing: {len(review.missing_investigations)}. "
                f"{'✓ APPROVED' if approved else '✗ REJECTED'}"
            ),
        }

        logger.info(f"[Critic] {'Approved' if approved else 'Rejected'}: quality={review.overall_quality}")

        return {
            "critic_review": review,
            "reanalysis_requests": reanalysis_requests,
            "agent_trace": [trace_entry],
            "current_node": next_node,
        }

    except json.JSONDecodeError as exc:
        logger.warning(f"[Critic] JSON parse error: {exc}, auto-approving")
        review = CriticReview(
            approved=True,
            overall_quality=70,
            invalid_findings=[],
            missing_investigations=[],
            required_reanalysis=[],
            duplicate_findings=[],
            comments=["Auto-approved due to critic response parse error."],
        )
        return {
            "critic_review": review,
            "agent_trace": [{
                "node": "Critic Agent",
                "iteration": revision_count + 1,
                "action": "Parse error — auto-approved",
                "result": "Findings approved (critic JSON error)",
            }],
            "current_node": "reporter",
        }

    except Exception as exc:
        logger.error(f"[Critic] Error: {exc}", exc_info=True)
        review = CriticReview(
            approved=True,
            overall_quality=60,
            invalid_findings=[],
            missing_investigations=[],
            required_reanalysis=[],
            duplicate_findings=[],
            comments=[f"Auto-approved due to critic error: {str(exc)[:100]}"],
        )
        return {
            "critic_review": review,
            "errors": [f"Critic Agent error: {str(exc)}"],
            "agent_trace": [{
                "node": "Critic Agent",
                "iteration": revision_count + 1,
                "action": "Error — auto-approved",
                "result": str(exc)[:200],
            }],
            "current_node": "reporter",
        }


# ---------------------------------------------------------------------------
# Conditional Edge Logic
# ---------------------------------------------------------------------------

def critic_routing(state: AnalysisState) -> str:
    """
    LangGraph conditional edge: determines where to route after critic review.

    Returns:
        "reporter" — findings approved, generate final report
        "re_investigate" — findings need more work
    """
    review = state.get("critic_review")
    revision_count = state.get("revision_count", 0)

    if revision_count >= MAX_REVISION_CYCLES:
        return "reporter"

    if review is None:
        return "reporter"

    return "reporter" if review.approved else "re_investigate"
