"""
Reporter Agent — CodePilot AI.

Generates the final professional software engineering report once
findings have been approved by the Critic.

Produces both a FinalReport Pydantic object and a full Markdown document.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from config import get_llm
from graph.state import AnalysisState, AgentTraceEntry
from models.schemas import Finding, FinalReport, FindingCategory, Severity

logger = logging.getLogger(__name__)

REPORT_PROMPT = """You are a senior software engineering lead writing a professional code review report.

Repository: {repo_name}
Languages: {languages}
Frameworks: {frameworks}
Total findings: {total_findings}

Findings by severity:
- Critical: {critical_count}
- High: {high_count}
- Medium: {medium_count}
- Low: {low_count}
- Info: {info_count}

Agent summaries:
{agent_summaries}

Write a comprehensive Executive Summary (3-4 paragraphs) for this engineering report.
The summary should:
1. Describe the overall state of the codebase
2. Highlight the most critical issues
3. Note the strengths of the codebase
4. Recommend next steps

Also provide:
- A list of 5-10 prioritized recommended actions (ordered from most to least urgent)
- A Repository Overview paragraph (2-3 sentences about the project's tech stack and structure)

Return ONLY a JSON object:
{{
  "executive_summary": "...",
  "repository_overview": "...",
  "recommended_actions": ["Action 1 (Critical)", "Action 2 (High)", ...]
}}
"""


def _calculate_scores(findings: List[Finding]) -> Dict[str, int]:
    """
    Calculate category scores based on findings.
    Higher score = better (fewer/lower severity issues).
    """
    def _category_score(category_findings: List[Finding]) -> int:
        if not category_findings:
            return 90  # No findings = good score
        penalty = sum({
            Severity.CRITICAL: 30,
            Severity.HIGH: 15,
            Severity.MEDIUM: 7,
            Severity.LOW: 3,
            Severity.INFO: 1,
        }.get(f.severity, 5) for f in category_findings)
        return max(10, min(95, 95 - penalty))

    security_findings = [f for f in findings if f.category == FindingCategory.SECURITY]
    code_findings = [f for f in findings if f.category in (FindingCategory.CODE_QUALITY, FindingCategory.BUG)]
    test_findings = [f for f in findings if f.category == FindingCategory.TESTING]
    arch_findings = [f for f in findings if f.category == FindingCategory.ARCHITECTURE]
    doc_findings = [f for f in findings if f.category == FindingCategory.DOCUMENTATION]

    security_score = _category_score(security_findings)
    code_score = _category_score(code_findings)
    test_score = _category_score(test_findings)
    arch_score = _category_score(arch_findings)
    doc_score = _category_score(doc_findings)

    # Overall health = weighted average
    health = int(
        security_score * 0.30 +
        code_score * 0.25 +
        test_score * 0.20 +
        arch_score * 0.15 +
        doc_score * 0.10
    )

    return {
        "health": health,
        "security": security_score,
        "code_quality": code_score,
        "testing": test_score,
        "architecture": arch_score,
        "documentation": doc_score,
    }


def _build_markdown_report(
    state: AnalysisState,
    scores: Dict[str, int],
    executive_summary: str,
    repo_overview: str,
    recommended_actions: List[str],
) -> str:
    """Build the full Markdown engineering report."""
    findings = state.get("all_findings", [])
    repo_summary = state.get("repo_summary")
    critic_review = state.get("critic_review")

    repo_name = (
        repo_summary.root_path.split("/")[-1].split("\\")[-1]
        if repo_summary else "Repository"
    )

    lines = [
        f"# 🧑💻 CodePilot AI — Engineering Report",
        f"## {repo_name}",
        "",
        "---",
        "",
        "## 📊 Repository Health Dashboard",
        "",
        f"| Metric | Score |",
        f"|--------|-------|",
        f"| 🏥 Overall Health | {scores['health']}/100 |",
        f"| 🔐 Security Risk | {scores['security']}/100 |",
        f"| 💻 Code Quality | {scores['code_quality']}/100 |",
        f"| 🧪 Testing Score | {scores['testing']}/100 |",
        f"| 🏗️ Architecture | {scores['architecture']}/100 |",
        f"| 📖 Documentation | {scores['documentation']}/100 |",
        "",
        "---",
        "",
        "## 📝 Executive Summary",
        "",
        executive_summary,
        "",
        "---",
        "",
        "## 🏛️ Repository Overview",
        "",
        repo_overview,
        "",
    ]

    if repo_summary:
        lines += [
            f"**Languages:** {', '.join(repo_summary.languages)}",
            f"**Frameworks:** {', '.join(repo_summary.frameworks) or 'None detected'}",
            f"**Total Files:** {repo_summary.total_files}",
            f"**Test Files:** {repo_summary.test_files}",
            f"**Has CI/CD:** {'Yes' if repo_summary.has_ci else 'No'}",
            "",
        ]

    # Critical findings
    critical = [f for f in findings if f.severity == Severity.CRITICAL]
    high = [f for f in findings if f.severity == Severity.HIGH]
    priority_findings = critical + high[:5]

    if priority_findings:
        lines += [
            "---",
            "",
            "## 🚨 Critical & High Priority Findings",
            "",
        ]
        for f in priority_findings:
            lines += [
                f"### {'🔴' if f.severity == Severity.CRITICAL else '🟠'} [{f.severity.value}] {f.title}",
                "",
                f"**Category:** {f.category.value}  ",
                f"**File:** `{f.file or 'Not specified'}`  ",
                f"**Line:** {f.line or 'N/A'}  ",
                f"**Confidence:** {f.confidence_pct()}%  ",
                f"**Source:** {f.source.value}",
                "",
                f"**Description:** {f.description}",
                "",
            ]
            if f.evidence:
                lines += [
                    "**Evidence:**",
                    "```",
                    f.evidence[:500],
                    "```",
                    "",
                ]
            lines += [
                f"**Recommendation:** {f.recommendation}",
                "",
                "---",
                "",
            ]

    # Security findings
    sec_findings = [f for f in findings if f.category == FindingCategory.SECURITY]
    if sec_findings:
        lines += [
            "## 🔐 Security Findings",
            "",
            "> ⚠️ **Disclaimer:** Findings are generated by pattern matching and LLM analysis.",
            "> Manual verification by a qualified security professional is required before",
            "> acting on any security finding. This tool does NOT replace professional security audits.",
            "",
        ]
        for f in sec_findings:
            severity_icon = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢", "Info": "ℹ️"}.get(f.severity.value, "•")
            lines += [
                f"### {severity_icon} {f.title}",
                f"**Severity:** {f.severity.value} | **File:** `{f.file or 'N/A'}` | **Line:** {f.line or 'N/A'} | **Confidence:** {f.confidence_pct()}%",
                "",
                f"{f.description}",
                "",
            ]
            if f.evidence:
                lines += ["```", f.evidence[:300], "```", ""]
            lines += [f"**Fix:** {f.recommendation}", ""]

    # Code quality findings
    code_findings = [f for f in findings if f.category in (FindingCategory.CODE_QUALITY, FindingCategory.BUG)]
    if code_findings:
        lines += ["## 💻 Code Quality Findings", ""]
        for f in code_findings:
            severity_icon = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢", "Info": "ℹ️"}.get(f.severity.value, "•")
            lines += [
                f"### {severity_icon} {f.title}",
                f"**Severity:** {f.severity.value} | **File:** `{f.file or 'N/A'}` | **Line:** {f.line or 'N/A'}",
                "",
                f"{f.description}",
                "",
            ]
            if f.evidence:
                lines += ["```", f.evidence[:200], "```", ""]
            lines += [f"**Fix:** {f.recommendation}", ""]

    # Testing findings
    test_findings = [f for f in findings if f.category == FindingCategory.TESTING]
    if test_findings:
        lines += ["## 🧪 Testing Findings", ""]
        for f in test_findings:
            lines += [
                f"### {f.title}",
                f"**Severity:** {f.severity.value} | **File:** `{f.file or 'N/A'}`",
                "",
                f"{f.description}",
                "",
                f"**Recommendation:** {f.recommendation}",
                "",
            ]

    # Architecture findings
    arch_findings = [f for f in findings if f.category == FindingCategory.ARCHITECTURE]
    if arch_findings:
        lines += ["## 🏗️ Architecture Findings", ""]
        for f in arch_findings:
            lines += [
                f"### {f.title}",
                f"**Severity:** {f.severity.value} | **File:** `{f.file or 'N/A'}`",
                "",
                f"{f.description}",
                "",
                f"**Recommendation:** {f.recommendation}",
                "",
            ]

    # Documentation findings
    doc_findings = [f for f in findings if f.category == FindingCategory.DOCUMENTATION]
    if doc_findings:
        lines += ["## 📖 Documentation Findings", ""]
        for f in doc_findings:
            lines += [
                f"### {f.title}",
                f"**Severity:** {f.severity.value} | **File:** `{f.file or 'N/A'}`",
                "",
                f"{f.description}",
                "",
                f"**Recommendation:** {f.recommendation}",
                "",
            ]

    # Recommended actions
    if recommended_actions:
        lines += [
            "---",
            "",
            "## 🎯 Recommended Actions",
            "",
            "_Ordered by priority:_",
            "",
        ]
        for i, action in enumerate(recommended_actions, 1):
            lines.append(f"{i}. {action}")
        lines.append("")

    # Critic review summary
    if critic_review:
        lines += [
            "---",
            "",
            "## 🔍 Review Process",
            "",
            f"**Quality Score:** {critic_review.overall_quality}/100",
            f"**Revision Cycles:** {state.get('revision_count', 0)}",
        ]
        if critic_review.comments:
            lines += ["**Critic Comments:**"]
            for comment in critic_review.comments:
                lines.append(f"- {comment}")
        lines.append("")

    lines += [
        "---",
        "",
        "_Report generated by CodePilot AI — Multi-Agent Software Engineering Assistant_",
        "_Powered by LangGraph + Groq_",
        "",
        "> ⚠️ **Important Disclaimer:** This report is generated by AI analysis and should be",
        "> treated as a starting point for manual code review. Not all findings may be valid.",
        "> Critical security issues should be validated by qualified security professionals.",
    ]

    return "\n".join(lines)


def run_reporter(state: AnalysisState) -> Dict:
    """
    LangGraph node: Reporter Agent.

    Generates the final professional engineering report.

    Args:
        state: Current graph state.

    Returns:
        State updates including the FinalReport.
    """
    findings = state.get("all_findings", [])
    revision_count = state.get("revision_count", 0)
    logger.info(f"[Reporter] Generating final report for {len(findings)} findings")

    repo_summary = state.get("repo_summary")
    repo_name = (
        repo_summary.root_path.split("/")[-1].split("\\")[-1]
        if repo_summary else "Repository"
    )

    # Calculate scores
    scores = _calculate_scores(findings)

    # Count by severity
    severity_counts = {s.value: 0 for s in Severity}
    for f in findings:
        severity_counts[f.severity.value] = severity_counts.get(f.severity.value, 0) + 1

    # Build agent summaries
    summaries = []
    for key, label in [
        ("code_review_summary", "Code Review"),
        ("security_summary", "Security"),
        ("testing_summary", "Testing"),
        ("architecture_summary", "Architecture"),
        ("documentation_summary", "Documentation"),
    ]:
        val = state.get(key, "")
        if val:
            summaries.append(f"{label}: {val}")

    # LLM executive summary
    executive_summary = "Analysis complete."
    repo_overview = ""
    recommended_actions = []

    try:
        llm = get_llm()
        prompt = REPORT_PROMPT.format(
            repo_name=repo_name,
            languages=", ".join(repo_summary.languages) if repo_summary else "Unknown",
            frameworks=", ".join(repo_summary.frameworks) if repo_summary else "Unknown",
            total_findings=len(findings),
            critical_count=severity_counts.get("Critical", 0),
            high_count=severity_counts.get("High", 0),
            medium_count=severity_counts.get("Medium", 0),
            low_count=severity_counts.get("Low", 0),
            info_count=severity_counts.get("Info", 0),
            agent_summaries="\n".join(summaries),
        )
        response = llm.invoke(prompt)
        content = response.content.strip()

        import re, json
        if "```" in content:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if match:
                content = match.group(1)

        report_data = json.loads(content)
        executive_summary = report_data.get("executive_summary", executive_summary)
        repo_overview = report_data.get("repository_overview", "")
        recommended_actions = report_data.get("recommended_actions", [])

    except Exception as exc:
        logger.warning(f"[Reporter] LLM error: {exc}, using fallback summary")
        executive_summary = (
            f"Analysis of {repo_name} identified {len(findings)} findings: "
            f"{severity_counts.get('Critical', 0)} Critical, "
            f"{severity_counts.get('High', 0)} High, "
            f"{severity_counts.get('Medium', 0)} Medium, "
            f"{severity_counts.get('Low', 0)} Low issues. "
            "Review the detailed findings below for specific recommendations."
        )
        recommended_actions = [
            f"Address {severity_counts.get('Critical', 0)} critical findings immediately",
            f"Fix {severity_counts.get('High', 0)} high severity issues",
            "Improve test coverage",
            "Review security findings with a qualified professional",
            "Update documentation",
        ]

    # Build full markdown report
    full_report_md = _build_markdown_report(
        state, scores, executive_summary, repo_overview, recommended_actions
    )

    final_report = FinalReport(
        executive_summary=executive_summary,
        health_score=scores["health"],
        security_risk_score=scores["security"],
        code_quality_score=scores["code_quality"],
        testing_score=scores["testing"],
        architecture_score=scores["architecture"],
        documentation_score=scores["documentation"],
        repository_overview=repo_overview or f"{repo_name} analyzed successfully.",
        critical_findings_summary=(
            f"{severity_counts.get('Critical', 0)} critical, "
            f"{severity_counts.get('High', 0)} high priority findings require immediate attention."
        ),
        recommended_actions=recommended_actions,
        full_report_markdown=full_report_md,
    )

    trace_entry: AgentTraceEntry = {
        "node": "Reporter",
        "iteration": revision_count + 1,
        "action": f"Generated final report for {len(findings)} findings",
        "result": f"Health score: {scores['health']}/100. Report ready.",
    }

    logger.info(f"[Reporter] Complete: health={scores['health']}/100")

    return {
        "final_report": final_report,
        "agent_trace": [trace_entry],
        "current_node": "done",
    }
