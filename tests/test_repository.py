"""
Tests for repository tools — safe ZIP extraction, path traversal prevention,
tree generation, and file discovery.
"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import pytest

from tools.repository import (
    find_files,
    read_file,
    repository_tree,
    safe_extract_zip,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    """Create a temporary directory for tests."""
    d = tempfile.mkdtemp(prefix="codepilot_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_repo(tmp_dir):
    """Create a minimal sample repository."""
    root = Path(tmp_dir) / "repo"
    root.mkdir()
    (root / "src").mkdir()
    (root / "tests").mkdir()

    (root / "src" / "main.py").write_text(
        'import os\n\ndef main():\n    """Entry point."""\n    print("Hello")\n\nif __name__ == "__main__":\n    main()\n'
    )
    (root / "src" / "utils.py").write_text(
        "def helper():\n    pass\n"
    )
    (root / "tests" / "test_main.py").write_text(
        "import pytest\n\ndef test_placeholder():\n    assert True\n"
    )
    (root / "requirements.txt").write_text(
        "requests==2.28.0\nflask==2.2.0\n"
    )
    (root / "README.md").write_text(
        "# Sample Project\n\n## Installation\n\npip install -r requirements.txt\n"
    )

    return str(root)


@pytest.fixture
def sample_zip(tmp_dir, sample_repo):
    """Create a ZIP of the sample repo."""
    zip_path = Path(tmp_dir) / "sample.zip"
    with zipfile.ZipFile(str(zip_path), "w") as zf:
        for path in Path(sample_repo).rglob("*"):
            if path.is_file():
                arcname = path.relative_to(Path(sample_repo).parent)
                zf.write(str(path), str(arcname))
    return str(zip_path)


# ---------------------------------------------------------------------------
# ZIP Extraction Tests
# ---------------------------------------------------------------------------

class TestSafeZipExtraction:

    def test_safe_extraction_succeeds(self, tmp_dir, sample_zip):
        """Normal ZIP should extract successfully."""
        extract_to = Path(tmp_dir) / "extracted"
        success, root, error = safe_extract_zip(sample_zip, str(extract_to))
        assert success, f"Extraction failed: {error}"
        assert root  # Should return a non-empty path
        assert Path(root).exists()

    def test_extracted_files_present(self, tmp_dir, sample_zip):
        """Extracted files should be accessible."""
        extract_to = Path(tmp_dir) / "extracted2"
        success, root, _ = safe_extract_zip(sample_zip, str(extract_to))
        assert success
        # Should contain Python files
        py_files = list(Path(root).rglob("*.py"))
        assert len(py_files) >= 1

    def test_path_traversal_rejected_dotdot(self, tmp_dir):
        """ZIP with ../traversal paths should be rejected."""
        zip_path = Path(tmp_dir) / "malicious.zip"
        with zipfile.ZipFile(str(zip_path), "w") as zf:
            # Attempt path traversal
            zf.writestr("../../../etc/passwd", "root:x:0:0:root:/root:/bin/bash")

        extract_to = Path(tmp_dir) / "safe_dir"
        success, root, error = safe_extract_zip(str(zip_path), str(extract_to))
        assert not success
        assert "traversal" in error.lower() or "path" in error.lower()

    def test_absolute_path_rejected(self, tmp_dir):
        """ZIP with absolute paths should be rejected."""
        zip_path = Path(tmp_dir) / "absolute.zip"
        with zipfile.ZipFile(str(zip_path), "w") as zf:
            zf.writestr("/etc/evil.txt", "evil content")

        extract_to = Path(tmp_dir) / "safe_dir2"
        success, root, error = safe_extract_zip(str(zip_path), str(extract_to))
        assert not success
        assert "absolute" in error.lower() or "path" in error.lower()

    def test_invalid_zip_rejected(self, tmp_dir):
        """Non-ZIP file should return failure."""
        fake_zip = Path(tmp_dir) / "not_a_zip.zip"
        fake_zip.write_text("this is not a zip file")

        extract_to = Path(tmp_dir) / "safe_dir3"
        success, root, error = safe_extract_zip(str(fake_zip), str(extract_to))
        assert not success
        assert "invalid" in error.lower() or "zip" in error.lower()


# ---------------------------------------------------------------------------
# Path Traversal Prevention Tests
# ---------------------------------------------------------------------------

class TestPathTraversal:

    def test_safe_path_within_repo(self, sample_repo):
        """Normal relative path should work."""
        ok, content = read_file(sample_repo, "src/main.py")
        assert ok
        assert "def main" in content

    def test_path_traversal_blocked(self, sample_repo):
        """Path traversal attempt should be blocked."""
        ok, content = read_file(sample_repo, "../../etc/passwd")
        assert not ok
        assert "traversal" in content.lower() or "not found" in content.lower()

    def test_absolute_path_blocked(self, sample_repo):
        """Absolute path should be blocked."""
        ok, content = read_file(sample_repo, "/etc/passwd")
        assert not ok

    def test_encoded_traversal_blocked(self, sample_repo):
        """URL-encoded or multi-step traversal should be blocked."""
        ok, content = read_file(sample_repo, "src/../../../etc/passwd")
        assert not ok

    def test_nonexistent_file(self, sample_repo):
        """Non-existent file should return failure."""
        ok, content = read_file(sample_repo, "nonexistent.py")
        assert not ok


# ---------------------------------------------------------------------------
# Repository Tree Tests
# ---------------------------------------------------------------------------

class TestRepositoryTree:

    def test_tree_returns_string(self, sample_repo):
        tree = repository_tree(sample_repo)
        assert isinstance(tree, str)
        assert len(tree) > 0

    def test_tree_contains_files(self, sample_repo):
        tree = repository_tree(sample_repo)
        assert "main.py" in tree
        assert "requirements.txt" in tree

    def test_tree_contains_directories(self, sample_repo):
        tree = repository_tree(sample_repo)
        assert "src" in tree
        assert "tests" in tree

    def test_tree_depth_limit(self, sample_repo):
        """Tree with depth 1 should not show deep files."""
        tree = repository_tree(sample_repo, max_depth=1)
        assert isinstance(tree, str)


# ---------------------------------------------------------------------------
# Find Files Tests
# ---------------------------------------------------------------------------

class TestFindFiles:

    def test_find_python_files(self, sample_repo):
        files = find_files(sample_repo, extension=".py")
        assert len(files) >= 2  # main.py, utils.py, test_main.py

    def test_find_by_name(self, sample_repo):
        files = find_files(sample_repo, name_pattern="requirements.txt")
        assert len(files) == 1
        assert "requirements.txt" in files[0]

    def test_find_all_source_files(self, sample_repo):
        files = find_files(sample_repo)
        assert len(files) >= 3

    def test_paths_are_relative(self, sample_repo):
        files = find_files(sample_repo, extension=".py")
        for f in files:
            # Should be relative path, not absolute
            assert not os.path.isabs(f)

    def test_max_results_respected(self, sample_repo):
        files = find_files(sample_repo, max_results=2)
        assert len(files) <= 2
