"""
Tests for LangGraph graph state, critic routing, revision counting,
and maximum iteration protection.

All LLM calls are mocked — no real API calls are made.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config import MAX_REVISION_CYCLES
from graph.state import AnalysisState
from models.schemas import (
    CriticReview,
    Finding,
    FindingCategory,
    FindingSource,
    InvestigationPlan,
    ReanalysisRequest,
    RepositorySummary,
    Severity,
)


# ---------------------------------------------------------------------------
# Helper: Create sample state
# ---------------------------------------------------------------------------

def make_sample_finding(
    title="Test Finding",
    severity=Severity.MEDIUM,
    category=FindingCategory.SECURITY,
    evidence="some_code_here",
) -> Finding:
    return Finding(
        category=category,
        title=title,
        severity=severity,
        file="src/test.py",
        line=10,
        description="Test description",
        evidence=evidence,
        recommendation="Fix it",
        confidence=0.8,
        source=FindingSource.DETERMINISTIC,
    )


def make_sample_state(
    revision_count: int = 0,
    findings=None,
    critic_review=None,
) -> AnalysisState:
    return {
        "repo_path": "/tmp/test_repo",
        "user_request": "Analyze this code",
        "focus_areas": ["security"],
        "repo_summary": RepositorySummary(
            root_path="/tmp/test_repo",
            languages=["Python"],
            frameworks=["Flask"],
            total_files=10,
            source_files=8,
            test_files=2,
            config_files=1,
            has_readme=True,
            has_tests=True,
            has_ci=False,
            entry_points=["app.py"],
            directory_tree="repo/\n└── src/",
        ),
        "investigation_plan": InvestigationPlan(
            focus_areas=["security"],
            agents_required=["security", "code_review"],
            rationale="Test plan",
        ),
        "all_findings": findings or [make_sample_finding()],
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
        "critic_review": critic_review,
        "revision_count": revision_count,
        "reanalysis_requests": [],
        "final_report": None,
        "agent_trace": [],
        "current_node": "critic",
        "errors": [],
    }


# ---------------------------------------------------------------------------
# Critic Routing Tests
# ---------------------------------------------------------------------------

class TestCriticRouting:

    def test_approved_routes_to_reporter(self):
        """Approved review should route to reporter."""
        from agents.critic import critic_routing

        state = make_sample_state(
            critic_review=CriticReview(
                approved=True,
                overall_quality=85,
            )
        )
        assert critic_routing(state) == "reporter"

    def test_rejected_routes_to_reinvestigate(self):
        """Rejected review should route to re_investigate."""
        from agents.critic import critic_routing

        state = make_sample_state(
            revision_count=0,
            critic_review=CriticReview(
                approved=False,
                overall_quality=40,
                required_reanalysis=["security"],
            )
        )
        assert critic_routing(state) == "re_investigate"

    def test_max_iterations_forces_reporter(self):
        """When revision_count >= MAX_REVISION_CYCLES, must route to reporter."""
        from agents.critic import critic_routing

        state = make_sample_state(
            revision_count=MAX_REVISION_CYCLES,
            critic_review=CriticReview(
                approved=False,  # Even if rejected...
                overall_quality=30,
            )
        )
        # Should still route to reporter due to max iterations
        assert critic_routing(state) == "reporter"

    def test_none_critic_review_routes_to_reporter(self):
        """Missing critic review should default to reporter."""
        from agents.critic import critic_routing

        state = make_sample_state(critic_review=None)
        assert critic_routing(state) == "reporter"


# ---------------------------------------------------------------------------
# Revision Count Tests
# ---------------------------------------------------------------------------

class TestRevisionCount:

    def test_max_revision_cycles_constant(self):
        """MAX_REVISION_CYCLES should be defined and reasonable."""
        assert isinstance(MAX_REVISION_CYCLES, int)
        assert 1 <= MAX_REVISION_CYCLES <= 10

    def test_revision_count_increments_in_reinvestigate(self):
        """Re-investigation should increment revision_count."""
        from graph.workflow import run_re_investigate

        state = make_sample_state(revision_count=0)

        with patch("graph.workflow.run_security_agent") as mock_sec:
            mock_sec.return_value = {
                "all_findings": [],
                "security_summary": "re-analyzed",
                "security_files": [],
                "agent_trace": [],
            }
            state["reanalysis_requests"] = [
                ReanalysisRequest(
                    agent_name="security",
                    reason="Test",
                    focus_areas=["auth"],
                )
            ]

            result = run_re_investigate(state)
            assert result["revision_count"] == 1

    def test_max_cycles_enforced_in_critic(self):
        """Critic should auto-approve when revision_count >= MAX_REVISION_CYCLES."""
        from agents.critic import run_critic

        state = make_sample_state(revision_count=MAX_REVISION_CYCLES)

        with patch("agents.critic.get_llm") as mock_llm:
            result = run_critic(state)
            # LLM should NOT be called (max iterations protection)
            mock_llm.assert_not_called()
            assert result["critic_review"].approved is True


# ---------------------------------------------------------------------------
# State Merge (Reducer) Tests
# ---------------------------------------------------------------------------

class TestStateMerge:

    def test_findings_reducer_deduplicates(self):
        """_merge_findings should deduplicate by title."""
        from graph.state import _merge_findings

        existing = [make_sample_finding(title="Finding A")]
        new = [
            make_sample_finding(title="Finding A"),  # Duplicate
            make_sample_finding(title="Finding B"),  # New
        ]
        merged = _merge_findings(existing, new)
        assert len(merged) == 2
        titles = [f.title for f in merged]
        assert "Finding A" in titles
        assert "Finding B" in titles

    def test_findings_reducer_appends_new(self):
        """_merge_findings should append non-duplicate findings."""
        from graph.state import _merge_findings

        existing = [make_sample_finding(title="Finding A")]
        new = [make_sample_finding(title="Finding C")]
        merged = _merge_findings(existing, new)
        assert len(merged) == 2

    def test_empty_merge(self):
        """Merging with empty lists should work."""
        from graph.state import _merge_findings

        existing = []
        new = [make_sample_finding(title="New Finding")]
        merged = _merge_findings(existing, new)
        assert len(merged) == 1


# ---------------------------------------------------------------------------
# Critic Agent Tests (Mocked LLM)
# ---------------------------------------------------------------------------

class TestCriticAgent:

    def test_critic_approves_with_good_findings(self):
        """Critic should approve when LLM returns approved=True."""
        from agents.critic import run_critic

        state = make_sample_state(revision_count=0)
        mock_response = MagicMock()
        mock_response.content = '{"approved": true, "overall_quality": 85, "invalid_findings": [], "missing_investigations": [], "required_reanalysis": [], "duplicate_findings": [], "comments": ["Good analysis"]}'

        with patch("agents.critic.get_llm") as mock_llm:
            mock_llm.return_value.invoke.return_value = mock_response
            result = run_critic(state)

        assert result["critic_review"].approved is True
        assert result["critic_review"].overall_quality == 85

    def test_critic_rejects_poor_findings(self):
        """Critic should reject when LLM returns approved=False."""
        from agents.critic import run_critic

        state = make_sample_state(revision_count=0)
        mock_response = MagicMock()
        mock_response.content = '{"approved": false, "overall_quality": 40, "invalid_findings": ["Test Finding"], "missing_investigations": ["auth module"], "required_reanalysis": [{"agent_name": "security", "reason": "Need more evidence", "focus_areas": ["auth"], "specific_files": []}], "duplicate_findings": [], "comments": ["Insufficient evidence"]}'

        with patch("agents.critic.get_llm") as mock_llm:
            mock_llm.return_value.invoke.return_value = mock_response
            result = run_critic(state)

        assert result["critic_review"].approved is False
        assert result["critic_review"].overall_quality == 40

    def test_critic_handles_json_parse_error(self):
        """Critic should auto-approve on JSON parse failure."""
        from agents.critic import run_critic

        state = make_sample_state(revision_count=0)
        mock_response = MagicMock()
        mock_response.content = "This is not valid JSON at all!!"

        with patch("agents.critic.get_llm") as mock_llm:
            mock_llm.return_value.invoke.return_value = mock_response
            result = run_critic(state)

        # Should auto-approve on parse error
        assert result["critic_review"].approved is True

    def test_critic_handles_llm_error(self):
        """Critic should auto-approve on LLM exception."""
        from agents.critic import run_critic

        state = make_sample_state(revision_count=0)

        with patch("agents.critic.get_llm") as mock_llm:
            mock_llm.return_value.invoke.side_effect = Exception("Network error")
            result = run_critic(state)

        # Should auto-approve on error
        assert result["critic_review"].approved is True
        assert len(result["errors"]) > 0


# ---------------------------------------------------------------------------
# Aggregator Tests
# ---------------------------------------------------------------------------

class TestFindingsAggregator:

    def test_aggregator_deduplicates(self):
        """Aggregator should remove near-duplicate findings."""
        from agents.aggregator import run_findings_aggregator

        findings = [
            make_sample_finding(title="SQL Injection", evidence="code1"),
            make_sample_finding(title="SQL Injection", evidence="code2"),  # Duplicate title
            make_sample_finding(title="XSS Vulnerability", evidence="code3"),
        ]
        state = make_sample_state(findings=findings)
        result = run_findings_aggregator(state)

        # All findings after aggregation
        output_findings = result.get("all_findings", findings)
        # Should have fewer than 3 (deduped)
        assert len([f for f in output_findings if "SQL Injection" in f.title]) <= 1

    def test_aggregator_sorts_by_severity(self):
        """Critical findings should come first."""
        from agents.aggregator import run_findings_aggregator

        findings = [
            make_sample_finding(title="Low Finding", severity=Severity.LOW),
            make_sample_finding(title="Critical Finding", severity=Severity.CRITICAL),
            make_sample_finding(title="High Finding", severity=Severity.HIGH),
        ]
        state = make_sample_state(findings=findings)
        result = run_findings_aggregator(state)

        output = result.get("all_findings", [])
        if len(output) >= 2:
            # First finding should be Critical or at least not Low
            assert output[0].severity != Severity.LOW or len(output) == 1

    def test_aggregator_adds_trace(self):
        """Aggregator should add an agent trace entry."""
        from agents.aggregator import run_findings_aggregator

        state = make_sample_state(findings=[make_sample_finding()])
        result = run_findings_aggregator(state)
        assert len(result.get("agent_trace", [])) > 0
