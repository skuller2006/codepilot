"""
Pydantic models for CodePilot AI structured outputs.
All findings, reviews, plans, and reports are typed here.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    INFO = "Info"


class FindingCategory(str, Enum):
    SECURITY = "Security"
    CODE_QUALITY = "Code Quality"
    TESTING = "Testing"
    ARCHITECTURE = "Architecture"
    DOCUMENTATION = "Documentation"
    PERFORMANCE = "Performance"
    BUG = "Bug"


class FindingSource(str, Enum):
    DETERMINISTIC = "Deterministic Tool"  # from static tool scanning
    LLM = "LLM Analysis"                  # from language model


# ---------------------------------------------------------------------------
# Core Finding Model
# ---------------------------------------------------------------------------

class Finding(BaseModel):
    """A single finding produced by any specialist agent."""

    category: FindingCategory
    title: str = Field(..., description="Short descriptive title")
    severity: Severity
    file: Optional[str] = Field(None, description="Relative file path within the repo")
    line: Optional[int] = Field(None, description="Line number, if available")
    description: str = Field(..., description="Full explanation of the finding")
    evidence: Optional[str] = Field(None, description="Code snippet or log evidence")
    recommendation: str = Field(..., description="Actionable recommendation")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0"
    )
    source: FindingSource = FindingSource.LLM

    def confidence_pct(self) -> int:
        return int(self.confidence * 100)


# ---------------------------------------------------------------------------
# Agent Outputs
# ---------------------------------------------------------------------------

class InvestigationPlan(BaseModel):
    """Produced by the Planning Agent."""

    focus_areas: List[str] = Field(
        ..., description="High-priority areas to investigate"
    )
    agents_required: List[str] = Field(
        ..., description="Which specialist agents should run"
    )
    rationale: str = Field(..., description="Why this plan was chosen")
    key_files_to_examine: List[str] = Field(
        default_factory=list,
        description="Specific files the agents should examine closely"
    )


class RepositorySummary(BaseModel):
    """Produced by the Repository Analyzer."""

    root_path: str
    languages: List[str]
    frameworks: List[str]
    total_files: int
    source_files: int
    test_files: int
    config_files: int
    has_readme: bool
    has_tests: bool
    has_ci: bool
    entry_points: List[str]
    dependencies: dict = Field(default_factory=dict)
    directory_tree: str = Field(..., description="Human-readable directory tree")
    key_findings: List[str] = Field(
        default_factory=list,
        description="Notable structural observations"
    )


class CriticReview(BaseModel):
    """Produced by the Critic Agent after reviewing all findings."""

    approved: bool
    overall_quality: int = Field(
        ..., ge=0, le=100, description="Quality score 0-100"
    )
    invalid_findings: List[str] = Field(
        default_factory=list,
        description="Titles of findings that lack sufficient evidence"
    )
    missing_investigations: List[str] = Field(
        default_factory=list,
        description="Important areas not yet investigated"
    )
    required_reanalysis: List[str] = Field(
        default_factory=list,
        description="Agent names that should re-run"
    )
    comments: List[str] = Field(
        default_factory=list,
        description="Detailed critic feedback comments"
    )
    duplicate_findings: List[str] = Field(
        default_factory=list,
        description="Titles of findings that duplicate other findings"
    )


class FinalReport(BaseModel):
    """The complete engineering report produced by the Reporter Agent."""

    executive_summary: str
    health_score: int = Field(..., ge=0, le=100)
    security_risk_score: int = Field(..., ge=0, le=100)
    code_quality_score: int = Field(..., ge=0, le=100)
    testing_score: int = Field(..., ge=0, le=100)
    architecture_score: int = Field(..., ge=0, le=100)
    documentation_score: int = Field(..., ge=0, le=100)
    repository_overview: str
    critical_findings_summary: str
    recommended_actions: List[str]
    full_report_markdown: str


# ---------------------------------------------------------------------------
# Agent Input/Output Wrappers
# ---------------------------------------------------------------------------

class AgentFindingsOutput(BaseModel):
    """Standard output wrapper for all specialist agents."""

    agent_name: str
    findings: List[Finding]
    summary: str
    files_analyzed: List[str] = Field(default_factory=list)


class ReanalysisRequest(BaseModel):
    """Sent from Critic back to specialist agents for re-investigation."""

    agent_name: str
    reason: str
    focus_areas: List[str]
    specific_files: List[str] = Field(default_factory=list)
