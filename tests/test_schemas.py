"""
Tests for Pydantic schemas and data models.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from models.schemas import (
    CriticReview,
    Finding,
    FindingCategory,
    FindingSource,
    FinalReport,
    InvestigationPlan,
    ReanalysisRequest,
    RepositorySummary,
    Severity,
)


class TestFindingModel:

    def test_valid_finding_creation(self):
        finding = Finding(
            category=FindingCategory.SECURITY,
            title="SQL Injection Risk",
            severity=Severity.HIGH,
            file="src/api.py",
            line=42,
            description="User input concatenated into SQL query",
            evidence="query = 'SELECT * FROM users WHERE id=' + user_id",
            recommendation="Use parameterized queries",
            confidence=0.95,
        )
        assert finding.title == "SQL Injection Risk"
        assert finding.severity == Severity.HIGH
        assert finding.confidence_pct() == 95

    def test_confidence_must_be_0_to_1(self):
        with pytest.raises(ValidationError):
            Finding(
                category=FindingCategory.SECURITY,
                title="Test",
                severity=Severity.LOW,
                description="Test",
                recommendation="Test",
                confidence=1.5,  # Invalid: > 1.0
            )

    def test_negative_confidence_rejected(self):
        with pytest.raises(ValidationError):
            Finding(
                category=FindingCategory.SECURITY,
                title="Test",
                severity=Severity.LOW,
                description="Test",
                recommendation="Test",
                confidence=-0.1,  # Invalid: < 0.0
            )

    def test_optional_fields_default_none(self):
        finding = Finding(
            category=FindingCategory.CODE_QUALITY,
            title="Missing docstring",
            severity=Severity.LOW,
            description="Function lacks documentation",
            recommendation="Add docstring",
            confidence=0.8,
        )
        assert finding.file is None
        assert finding.line is None
        assert finding.evidence is None

    def test_default_source_is_llm(self):
        finding = Finding(
            category=FindingCategory.SECURITY,
            title="Test",
            severity=Severity.INFO,
            description="Test",
            recommendation="Test",
            confidence=0.5,
        )
        assert finding.source == FindingSource.LLM

    def test_deterministic_source(self):
        finding = Finding(
            category=FindingCategory.SECURITY,
            title="Test",
            severity=Severity.INFO,
            description="Test",
            recommendation="Test",
            confidence=0.9,
            source=FindingSource.DETERMINISTIC,
        )
        assert finding.source == FindingSource.DETERMINISTIC

    def test_all_severity_values(self):
        for sev in Severity:
            finding = Finding(
                category=FindingCategory.SECURITY,
                title=f"Test {sev.value}",
                severity=sev,
                description="Test",
                recommendation="Test",
                confidence=0.7,
            )
            assert finding.severity == sev

    def test_all_categories(self):
        for cat in FindingCategory:
            finding = Finding(
                category=cat,
                title="Test",
                severity=Severity.LOW,
                description="Test",
                recommendation="Test",
                confidence=0.5,
            )
            assert finding.category == cat


class TestCriticReviewModel:

    def test_approved_review(self):
        review = CriticReview(
            approved=True,
            overall_quality=85,
            invalid_findings=[],
            missing_investigations=[],
            required_reanalysis=[],
            comments=["Good analysis"],
        )
        assert review.approved is True
        assert review.overall_quality == 85

    def test_rejected_review(self):
        review = CriticReview(
            approved=False,
            overall_quality=45,
            invalid_findings=["Unsupported claim 1"],
            missing_investigations=["Authentication flow"],
            required_reanalysis=["security"],
            comments=["Need more evidence"],
        )
        assert review.approved is False
        assert len(review.invalid_findings) == 1

    def test_quality_bounds(self):
        """Quality must be 0-100."""
        with pytest.raises(ValidationError):
            CriticReview(
                approved=True,
                overall_quality=150,  # Invalid
            )

    def test_quality_negative_rejected(self):
        with pytest.raises(ValidationError):
            CriticReview(
                approved=True,
                overall_quality=-5,  # Invalid
            )

    def test_default_empty_lists(self):
        review = CriticReview(
            approved=True,
            overall_quality=80,
        )
        assert review.invalid_findings == []
        assert review.missing_investigations == []
        assert review.comments == []


class TestInvestigationPlanModel:

    def test_valid_plan(self):
        plan = InvestigationPlan(
            focus_areas=["authentication", "database"],
            agents_required=["security", "code_review"],
            rationale="Security-focused analysis requested",
            key_files_to_examine=["src/auth.py", "src/models.py"],
        )
        assert len(plan.focus_areas) == 2
        assert "security" in plan.agents_required

    def test_empty_key_files_default(self):
        plan = InvestigationPlan(
            focus_areas=["general"],
            agents_required=["code_review"],
            rationale="Standard review",
        )
        assert plan.key_files_to_examine == []


class TestRepositorySummaryModel:

    def test_valid_summary(self):
        summary = RepositorySummary(
            root_path="/tmp/repo",
            languages=["Python", "JavaScript"],
            frameworks=["Flask", "React"],
            total_files=50,
            source_files=40,
            test_files=10,
            config_files=5,
            has_readme=True,
            has_tests=True,
            has_ci=False,
            entry_points=["app.py"],
            directory_tree="repo/\n├── src/\n└── tests/",
        )
        assert summary.has_readme is True
        assert summary.total_files == 50


class TestFinalReportModel:

    def test_valid_report(self):
        report = FinalReport(
            executive_summary="Good codebase overall.",
            health_score=75,
            security_risk_score=60,
            code_quality_score=80,
            testing_score=70,
            architecture_score=75,
            documentation_score=65,
            repository_overview="Python Flask application.",
            critical_findings_summary="2 critical findings.",
            recommended_actions=["Fix SQL injection", "Add tests"],
            full_report_markdown="# Report\n...",
        )
        assert report.health_score == 75
        assert len(report.recommended_actions) == 2

    def test_score_bounds_enforced(self):
        with pytest.raises(ValidationError):
            FinalReport(
                executive_summary="Test",
                health_score=110,  # Invalid: > 100
                security_risk_score=50,
                code_quality_score=50,
                testing_score=50,
                architecture_score=50,
                documentation_score=50,
                repository_overview="Test",
                critical_findings_summary="None",
                recommended_actions=[],
                full_report_markdown="",
            )


class TestReanalysisRequestModel:

    def test_valid_request(self):
        req = ReanalysisRequest(
            agent_name="security",
            reason="Insufficient evidence for SQL injection claim",
            focus_areas=["database queries", "ORM usage"],
            specific_files=["src/models.py", "src/api.py"],
        )
        assert req.agent_name == "security"
        assert len(req.focus_areas) == 2
