"""
Tests for code search and dependency analysis tools.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from tools.code_search import search_code, search_for_pattern_in_file
from tools.dependencies import dependency_analyzer
from tools.secret_scanner import secret_scanner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def code_repo(tmp_path):
    """Create a repo with various code patterns for testing."""
    src = tmp_path / "src"
    src.mkdir()

    (src / "auth.py").write_text(
        'import hashlib\n\n'
        'DB_PASSWORD = "super_secret_123"\n'
        'API_KEY = "sk_live_abc123xyz456789"\n\n'
        'def authenticate(user, password):\n'
        '    hashed = hashlib.md5(password.encode()).hexdigest()\n'
        '    return hashed\n\n'
        'def login(username, password):\n'
        '    query = "SELECT * FROM users WHERE name=\'" + username + "\'"\n'
        '    return query\n'
    )

    (src / "utils.py").write_text(
        'import os\n\n'
        'def run_command(cmd):\n'
        '    return os.system(cmd)\n\n'
        'def safe_function():\n'
        '    """This is well-documented."""\n'
        '    return 42\n'
    )

    (tmp_path / "requirements.txt").write_text(
        "requests==2.28.0\n"
        "flask==2.2.0\n"
        "sqlalchemy>=1.4.0\n"
        "# test comment\n"
        "pytest==7.2.0\n"
    )

    (tmp_path / "package.json").write_text(
        '{"name": "test-app", "version": "1.0.0", '
        '"dependencies": {"express": "^4.18.0", "lodash": "^4.17.0"}, '
        '"devDependencies": {"jest": "^29.0.0"}}'
    )

    return str(tmp_path)


# ---------------------------------------------------------------------------
# Code Search Tests
# ---------------------------------------------------------------------------

class TestCodeSearch:

    def test_search_finds_literal(self, code_repo):
        """Simple literal search should find matches."""
        results = search_code(code_repo, "def authenticate")
        assert len(results) > 0
        assert any(r["file"].endswith("auth.py") for r in results)

    def test_search_returns_line_numbers(self, code_repo):
        """Results should include line numbers."""
        results = search_code(code_repo, "os.system")
        assert len(results) > 0
        assert all("line_number" in r for r in results)
        assert all(isinstance(r["line_number"], int) for r in results)

    def test_regex_search(self, code_repo):
        """Regex search should work."""
        results = search_code(code_repo, r"def \w+\(", is_regex=True)
        assert len(results) >= 3  # multiple function defs

    def test_max_results_respected(self, code_repo):
        """max_results should be respected."""
        results = search_code(code_repo, "def", max_results=2)
        assert len(results) <= 2

    def test_search_specific_extension(self, code_repo):
        """Extension filter should limit search scope."""
        results = search_code(code_repo, "def", file_extensions=[".py"])
        assert all(r["file"].endswith(".py") for r in results if not r.get("error"))

    def test_context_lines_included(self, code_repo):
        """Context lines should surround each match."""
        results = search_code(code_repo, "hashlib", context_lines=2)
        assert len(results) > 0
        assert "context" in results[0]

    def test_invalid_regex_returns_error(self, code_repo):
        """Invalid regex should return error dict."""
        results = search_code(code_repo, "[invalid(regex", is_regex=True)
        assert len(results) == 1
        assert "error" in results[0]

    def test_search_in_single_file(self, code_repo):
        """search_for_pattern_in_file should work."""
        results = search_for_pattern_in_file(code_repo, "src/auth.py", "hashlib")
        assert len(results) > 0
        assert results[0]["line_number"] > 0


# ---------------------------------------------------------------------------
# Dependency Analyzer Tests
# ---------------------------------------------------------------------------

class TestDependencyAnalyzer:

    def test_finds_requirements_txt(self, code_repo):
        result = dependency_analyzer(code_repo)
        assert "requirements.txt" in result["manifests_found"]

    def test_finds_package_json(self, code_repo):
        result = dependency_analyzer(code_repo)
        assert "package.json" in result["manifests_found"]

    def test_parses_requirements_txt(self, code_repo):
        result = dependency_analyzer(code_repo)
        dep_names = [d["name"] for d in result["dependencies"]]
        assert "requests" in dep_names
        assert "flask" in dep_names

    def test_parses_package_json(self, code_repo):
        result = dependency_analyzer(code_repo)
        dep_names = [d["name"] for d in result["dependencies"]]
        assert "express" in dep_names
        assert "lodash" in dep_names

    def test_total_count_correct(self, code_repo):
        result = dependency_analyzer(code_repo)
        assert result["total_count"] == len(result["dependencies"])

    def test_empty_repo_returns_empty(self, tmp_path):
        """Empty repo should return empty results."""
        result = dependency_analyzer(str(tmp_path))
        assert result["total_count"] == 0
        assert result["manifests_found"] == []


# ---------------------------------------------------------------------------
# Secret Scanner Tests
# ---------------------------------------------------------------------------

class TestSecretScanner:

    def test_finds_hardcoded_password(self, code_repo):
        result = secret_scanner(code_repo)
        patterns = [f["pattern_name"] for f in result["findings"]]
        # Should find some secret pattern
        assert len(result["findings"]) > 0

    def test_disclaimer_present(self, code_repo):
        result = secret_scanner(code_repo)
        assert "disclaimer" in result
        assert "BASIC PORTFOLIO SCANNER" in result["disclaimer"]

    def test_files_scanned_count(self, code_repo):
        result = secret_scanner(code_repo)
        assert result["files_scanned"] > 0

    def test_placeholder_detection(self, tmp_path):
        """Placeholder values should be flagged as likely non-secrets."""
        (tmp_path / "config.py").write_text(
            'API_KEY = "your_api_key_here"\n'
            'SECRET = "replace_me_with_real_secret"\n'
        )
        result = secret_scanner(str(tmp_path))
        # High confidence findings exclude placeholders
        placeholder_findings = [
            f for f in result["findings"]
            if f["is_likely_placeholder"]
        ]
        assert len(placeholder_findings) > 0

    def test_empty_repo_no_findings(self, tmp_path):
        """Empty repo should have no findings."""
        result = secret_scanner(str(tmp_path))
        assert result["total_findings"] == 0
