"""
Security Agent — CodePilot AI.

Analyzes code for security vulnerabilities including injection attacks,
insecure authentication, hardcoded secrets, and dangerous patterns.

⚠ DISCLAIMER: This is a portfolio demonstration tool.
It is NOT a substitute for professional security audits, SAST tools,
or penetration testing. All findings must be manually verified.
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
from tools.code_search import search_code
from tools.secret_scanner import secret_scanner

logger = logging.getLogger(__name__)

SECURITY_PROMPT = """You are a security engineer performing a static code security review.

⚠ IMPORTANT: Only report findings you have DIRECT CODE EVIDENCE for.
Do NOT speculate or invent vulnerabilities. Do NOT claim a vulnerability
without a specific code snippet showing the problem.

{critic_feedback}

Repository context:
- Languages: {languages}
- Frameworks: {frameworks}
- Focus: {focus_areas}

Source files to analyze:
{code_context}

Deterministic scan results:
{scan_results}

Look for REAL security issues with code evidence:
1. SQL Injection — string concatenation in DB queries, lack of parameterization
2. Command Injection — subprocess/os.system with user input
3. Path Traversal — user-controlled file paths without validation
4. Authentication issues — weak passwords, missing auth checks, hardcoded credentials
5. Authorization failures — missing permission checks, IDOR vulnerabilities  
6. Insecure deserialization — pickle.loads, yaml.load (not safe_load), eval
7. Cross-Site Scripting — unsanitized user input in HTML output
8. Sensitive data exposure — logging passwords, PII in errors
9. Cryptographic weaknesses — MD5/SHA1 for passwords, random instead of secrets
10. SSRF — user-controlled URLs in server-side requests
11. XXE — XML parsing of untrusted input
12. Insecure direct object references

Return a JSON array:
[
  {{
    "title": "Specific vulnerability title",
    "severity": "Critical|High|Medium|Low|Info",
    "file": "relative/path/to/file.py",
    "line": 42,
    "description": "Detailed explanation of the security risk and attack vector",
    "evidence": "The exact vulnerable code snippet",
    "recommendation": "Specific fix with code example if possible",
    "confidence": 0.85
  }}
]

Return ONLY the JSON array.
"""


def _run_deterministic_security_scans(repo_root: str) -> tuple[List[Finding], str]:
    """
    Run deterministic security pattern searches.
    Returns (findings, summary_text).
    """
    findings: List[Finding] = []
    summary_parts = []

    # Secret scanner
    secret_results = secret_scanner(repo_root)
    high_conf_secrets = [
        f for f in secret_results["findings"]
        if not f["is_likely_placeholder"]
    ]
    if high_conf_secrets:
        summary_parts.append(
            f"Secret scanner: {len(high_conf_secrets)} potential secrets in "
            f"{secret_results['files_scanned']} files"
        )
        for secret in high_conf_secrets[:5]:
            findings.append(Finding(
                category=FindingCategory.SECURITY,
                title=f"Potential hardcoded secret: {secret['pattern_name']}",
                severity=Severity(secret["severity"]),
                file=secret["file"],
                line=secret["line_number"],
                description=(
                    f"A pattern matching '{secret['pattern_name']}' was found. "
                    "Hardcoded secrets can be exposed through source control, logs, or error messages."
                ),
                evidence=secret["line_content"][:200],
                recommendation=(
                    "Move secrets to environment variables. "
                    "Use a secrets manager (e.g., HashiCorp Vault, AWS Secrets Manager). "
                    "If this is a real secret, rotate it immediately and remove from git history."
                ),
                confidence=0.80,
                source=FindingSource.DETERMINISTIC,
            ))

    # Dangerous pattern searches
    dangerous_patterns = [
        ("SQL string concatenation", r'(?i)(execute|query)\s*\(\s*["\'].*\+|f".*SELECT|f\'.*SELECT', True, Severity.HIGH),
        ("subprocess shell=True", r"subprocess\.(call|run|Popen).*shell=True", True, Severity.HIGH),
        ("os.system usage", r"os\.system\s*\(", True, Severity.MEDIUM),
        ("pickle.loads", r"pickle\.loads?\s*\(", True, Severity.HIGH),
        ("yaml.load (unsafe)", r"yaml\.load\s*\([^,)]*\)", True, Severity.HIGH),
        ("eval with dynamic input", r"\beval\s*\(", True, Severity.HIGH),
        ("exec with dynamic input", r"\bexec\s*\(", True, Severity.HIGH),
        ("MD5 for passwords", r"(?i)(md5|sha1)\s*\(", True, Severity.MEDIUM),
        ("assert used for security", r"^\s*assert\s+", True, Severity.LOW),
        ("open redirect", r"redirect\s*\([^)]*request\.(args|form|get)", True, Severity.MEDIUM),
    ]

    for label, pattern, is_regex, severity in dangerous_patterns:
        results = search_code(repo_root, pattern, is_regex=is_regex, max_results=3)
        valid = [r for r in results if not r.get("error")]
        if valid:
            summary_parts.append(f"{label}: {len(valid)} occurrence(s)")
            for result in valid[:2]:
                findings.append(Finding(
                    category=FindingCategory.SECURITY,
                    title=f"Security risk: {label}",
                    severity=severity,
                    file=result["file"],
                    line=result["line_number"],
                    description=f"Detected usage of '{label}' which may introduce security vulnerabilities.",
                    evidence=result["line_content"][:300],
                    recommendation=f"Review usage of {label}. Consider safer alternatives.",
                    confidence=0.75,
                    source=FindingSource.DETERMINISTIC,
                ))

    return findings, "\n".join(summary_parts) or "No obvious patterns detected"


def run_security_agent(state: AnalysisState) -> Dict:
    """
    LangGraph node: Security Agent.

    Args:
        state: Current graph state.

    Returns:
        State updates dict.
    """
    repo_root = state["repo_path"]
    revision_count = state.get("revision_count", 0)
    logger.info(f"[Security] Running (revision {revision_count})")

    reanalysis_files, critic_feedback = get_reanalysis_context(state, "security")

    preferred_extensions = [".py", ".js", ".ts", ".java", ".go", ".php", ".rb", ".cs", ".env"]
    files = select_files_for_agent(
        state,
        preferred_extensions=preferred_extensions,
        reanalysis_files=reanalysis_files,
    )

    # Run deterministic scans first
    det_findings, scan_summary = _run_deterministic_security_scans(repo_root)

    if files:
        file_contents = read_files_for_analysis(repo_root, files)
        code_context = build_code_context(file_contents)
    else:
        code_context = "No source files available."

    repo_summary = state.get("repo_summary")
    languages = ", ".join(repo_summary.languages) if repo_summary else "Unknown"
    frameworks = ", ".join(repo_summary.frameworks) if repo_summary else "Unknown"
    plan = state.get("investigation_plan")
    focus_areas = ", ".join(plan.focus_areas) if plan else "security"

    prompt = SECURITY_PROMPT.format(
        critic_feedback=critic_feedback or "",
        languages=languages,
        frameworks=frameworks,
        focus_areas=focus_areas,
        code_context=code_context,
        scan_results=scan_summary,
    )

    llm_findings = []
    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        llm_findings = parse_findings_from_llm(
            response.content,
            FindingCategory.SECURITY,
            "Security",
        )
    except Exception as exc:
        logger.error(f"[Security] LLM error: {exc}", exc_info=True)

    all_findings = det_findings + llm_findings
    summary = (
        f"Security analysis of {len(files)} files: "
        f"{len(det_findings)} deterministic findings, "
        f"{len(llm_findings)} LLM-identified findings."
    )

    trace_entry: AgentTraceEntry = {
        "node": "Security Agent",
        "iteration": revision_count + 1,
        "action": f"Analyzed {len(files)} files + ran secret scanner",
        "result": f"Found {len(all_findings)} security issues ({len(det_findings)} deterministic)",
    }

    logger.info(f"[Security] Complete: {len(all_findings)} findings")

    return {
        "all_findings": all_findings,
        "security_summary": summary,
        "security_files": files,
        "agent_trace": [trace_entry],
    }
