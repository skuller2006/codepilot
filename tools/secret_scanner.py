"""
Secret Scanner — deterministic pattern-based secret detection.

⚠ IMPORTANT DISCLAIMER ⚠
This is a basic portfolio-grade secret scanner using pattern matching.
It is NOT a replacement for professional tools such as:
  - GitGuardian, TruffleHog, detect-secrets, git-secrets, Gitleaks

False positives are expected. False negatives are likely.
Do NOT rely on this scanner for production security audits.

This scanner intentionally avoids executing any code. It performs
purely static text-based pattern matching.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from tools.repository import find_files, read_file, SOURCE_EXTENSIONS


# ---------------------------------------------------------------------------
# Secret Patterns
# ---------------------------------------------------------------------------

SECRET_PATTERNS: List[Dict] = [
    {
        "name": "Generic API Key",
        "pattern": r'(?i)(api[_\-]?key|apikey)\s*[=:]\s*["\']?([A-Za-z0-9\-_]{20,})["\']?',
        "severity": "High",
    },
    {
        "name": "Generic Secret/Password",
        "pattern": r'(?i)(secret|password|passwd|pwd)\s*[=:]\s*["\']?([^\s"\'{}\[\]<>]{8,})["\']?',
        "severity": "High",
    },
    {
        "name": "Generic Token",
        "pattern": r'(?i)(token|auth_token|access_token|bearer)\s*[=:]\s*["\']?([A-Za-z0-9\-_.]{20,})["\']?',
        "severity": "High",
    },
    {
        "name": "AWS Access Key ID",
        "pattern": r'(?<![A-Z0-9])(AKIA[0-9A-Z]{16})(?![A-Z0-9])',
        "severity": "Critical",
    },
    {
        "name": "AWS Secret Access Key",
        "pattern": r'(?i)aws[_\-]?secret[_\-]?(access[_\-]?)?key\s*[=:]\s*["\']?([A-Za-z0-9+/]{40})["\']?',
        "severity": "Critical",
    },
    {
        "name": "Private Key Block",
        "pattern": r'-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----',
        "severity": "Critical",
    },
    {
        "name": "GitHub Token",
        "pattern": r'gh[ps]_[A-Za-z0-9]{36}',
        "severity": "Critical",
    },
    {
        "name": "Slack Token",
        "pattern": r'xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24,}',
        "severity": "High",
    },
    {
        "name": "Stripe API Key",
        "pattern": r'sk_(test|live)_[A-Za-z0-9]{24,}',
        "severity": "Critical",
    },
    {
        "name": "Google API Key",
        "pattern": r'AIza[0-9A-Za-z\-_]{35}',
        "severity": "High",
    },
    {
        "name": "Hardcoded Database URL",
        "pattern": r'(?i)(database[_\-]?url|db[_\-]?url|connection[_\-]?string)\s*[=:]\s*["\']?(postgresql|mysql|mongodb|sqlite)\+?://[^\s"\']+["\']?',
        "severity": "High",
    },
    {
        "name": "Hardcoded Database Password",
        "pattern": r'(?i)(db|database)[_\-]?pass(word)?\s*[=:]\s*["\']?([^\s"\'{}<>]{6,})["\']?',
        "severity": "High",
    },
    {
        "name": "JWT Secret",
        "pattern": r'(?i)(jwt[_\-]?secret|jwt[_\-]?key)\s*[=:]\s*["\']?([^\s"\'{}<>]{8,})["\']?',
        "severity": "High",
    },
    {
        "name": "SendGrid API Key",
        "pattern": r'SG\.[A-Za-z0-9\-_]{22}\.[A-Za-z0-9\-_]{43}',
        "severity": "High",
    },
    {
        "name": "Twilio Token",
        "pattern": r'(?i)(twilio[_\-]?auth[_\-]?token|TWILIO_AUTH_TOKEN)\s*[=:]\s*["\']?([a-f0-9]{32})["\']?',
        "severity": "High",
    },
    {
        "name": "Hardcoded IP Address (Potential)",
        "pattern": r'(?i)(host|server|endpoint)\s*[=:]\s*["\']?(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})["\']?',
        "severity": "Low",
    },
]

# Files and directories to exclude from scanning
EXCLUDE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".pdf",
    ".mp3", ".mp4", ".zip", ".tar", ".gz", ".pyc", ".so",
    ".min.js",  # minified JS tends to have high false-positive rate
}

EXCLUDE_FILENAMES = {
    ".gitignore", "LICENSE", "CHANGELOG.md", "CHANGELOG.rst",
}

EXCLUDE_DIRS = {
    "__pycache__", "node_modules", "venv", ".venv", "dist", "build",
    ".git", "migrations",
}

# Patterns that often indicate a placeholder, not a real secret
PLACEHOLDER_PATTERNS = re.compile(
    r'(?i)(example|placeholder|your[_\-]?(api[_\-]?)?key|changeme|'
    r'replace[_\-]?me|xxx+|<[^>]+>|\$\{[^}]+\}|\{\{[^}]+\}\}|'
    r'secret[_\-]?here|dummy|fake|test[_\-]?key|sample)',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class SecretFinding:
    """Represents a single potential secret found in the codebase."""

    def __init__(
        self,
        pattern_name: str,
        severity: str,
        file: str,
        line_number: int,
        line_content: str,
        matched_value: str,
        is_likely_placeholder: bool,
    ):
        self.pattern_name = pattern_name
        self.severity = severity
        self.file = file
        self.line_number = line_number
        self.line_content = line_content
        self.matched_value = matched_value
        self.is_likely_placeholder = is_likely_placeholder

    def to_dict(self) -> Dict:
        return {
            "pattern_name": self.pattern_name,
            "severity": self.severity,
            "file": self.file,
            "line_number": self.line_number,
            "line_content": self.line_content[:200],
            "matched_value": self.matched_value[:50] + "..." if len(self.matched_value) > 50 else self.matched_value,
            "is_likely_placeholder": self.is_likely_placeholder,
        }


def secret_scanner(repo_root: str) -> Dict:
    """
    ⚠ BASIC PORTFOLIO SCANNER — NOT FOR PRODUCTION USE ⚠

    Perform deterministic pattern-based scanning for potential secrets.

    Args:
        repo_root: Repository root path.

    Returns:
        {
            "disclaimer": str,
            "findings": List[Dict],
            "total_findings": int,
            "high_confidence_findings": int,
            "files_scanned": int,
        }
    """
    findings: List[SecretFinding] = []
    files_scanned = 0

    # Collect files to scan (source + config files)
    scan_extensions = SOURCE_EXTENSIONS | {".env", ".cfg", ".ini", ".conf", ".config"}
    all_files: List[str] = []
    for ext in scan_extensions:
        all_files.extend(find_files(repo_root, extension=ext))

    # Also check .env.example and similar
    from tools.repository import find_files as ff
    for special in [".env.example", ".env.sample"]:
        special_files = ff(repo_root, name_pattern=special)
        all_files.extend(special_files)

    all_files = list(dict.fromkeys(all_files))  # deduplicate

    compiled_patterns = [
        (p["name"], p["severity"], re.compile(p["pattern"], re.IGNORECASE))
        for p in SECRET_PATTERNS
    ]

    for rel_path in all_files:
        path = Path(rel_path)

        # Skip excluded
        if path.suffix.lower() in EXCLUDE_EXTENSIONS:
            continue
        if path.name in EXCLUDE_FILENAMES:
            continue
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue

        ok, content = read_file(repo_root, rel_path)
        if not ok:
            continue

        files_scanned += 1
        lines = content.splitlines()

        for line_idx, line in enumerate(lines):
            # Skip comment lines
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("//") or stripped.startswith("*"):
                continue

            for pattern_name, severity, regex in compiled_patterns:
                matches = regex.finditer(line)
                for match in matches:
                    matched_value = match.group(0)
                    is_placeholder = bool(PLACEHOLDER_PATTERNS.search(matched_value))

                    findings.append(SecretFinding(
                        pattern_name=pattern_name,
                        severity=severity,
                        file=rel_path,
                        line_number=line_idx + 1,
                        line_content=line.strip(),
                        matched_value=matched_value,
                        is_likely_placeholder=is_placeholder,
                    ))

    # Deduplicate by (file, line, pattern)
    seen = set()
    unique_findings = []
    for f in findings:
        key = (f.file, f.line_number, f.pattern_name)
        if key not in seen:
            seen.add(key)
            unique_findings.append(f)

    high_confidence = [f for f in unique_findings if not f.is_likely_placeholder]

    return {
        "disclaimer": (
            "⚠ BASIC PORTFOLIO SCANNER: This is a pattern-based heuristic scanner "
            "for demonstration purposes only. It is NOT a professional security tool. "
            "Expect false positives and false negatives. "
            "For production use, employ dedicated tools like GitGuardian, TruffleHog, or Gitleaks."
        ),
        "findings": [f.to_dict() for f in unique_findings],
        "high_confidence_findings": len(high_confidence),
        "total_findings": len(unique_findings),
        "files_scanned": files_scanned,
    }
