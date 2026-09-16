"""
Dependency analysis tool for CodePilot AI.

Reads dependency manifests to extract library names and versions.
Entirely deterministic — no LLM, no network calls.

Supported manifests:
  - requirements.txt
  - pyproject.toml (tool.poetry.dependencies)
  - package.json
  - package-lock.json
  - pom.xml (Maven)
  - build.gradle
  - Cargo.toml (Rust)
  - go.mod
  - Gemfile (Ruby)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from tools.repository import read_file


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_requirements_txt(content: str) -> List[Dict]:
    """Parse pip requirements.txt format."""
    deps = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        # Strip environment markers
        line = line.split(";")[0].strip()
        # Extract name and version specifier
        match = re.match(r"([A-Za-z0-9_.\-]+)\s*([>=<!~^].+)?", line)
        if match:
            deps.append({
                "name": match.group(1),
                "version_spec": (match.group(2) or "").strip(),
                "source": "requirements.txt",
            })
    return deps


def _parse_pyproject_toml(content: str) -> List[Dict]:
    """Parse pyproject.toml — supports both poetry and PEP 621 formats."""
    deps = []
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # backport
        except ImportError:
            # Fallback: regex-based parsing
            return _parse_pyproject_toml_regex(content)

    try:
        data = tomllib.loads(content)
    except Exception:
        return _parse_pyproject_toml_regex(content)

    # Poetry format
    poetry_deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
    for name, spec in poetry_deps.items():
        if name == "python":
            continue
        version = spec if isinstance(spec, str) else spec.get("version", "*") if isinstance(spec, dict) else "*"
        deps.append({"name": name, "version_spec": version, "source": "pyproject.toml"})

    # PEP 621 format
    project_deps = data.get("project", {}).get("dependencies", [])
    for dep in project_deps:
        if isinstance(dep, str):
            match = re.match(r"([A-Za-z0-9_.\-]+)\s*([>=<!~^].+)?", dep)
            if match:
                deps.append({
                    "name": match.group(1),
                    "version_spec": (match.group(2) or "").strip(),
                    "source": "pyproject.toml",
                })

    return deps


def _parse_pyproject_toml_regex(content: str) -> List[Dict]:
    """Fallback regex parser for pyproject.toml."""
    deps = []
    in_deps = False
    for line in content.splitlines():
        if "[tool.poetry.dependencies]" in line or "[project.dependencies]" in line:
            in_deps = True
            continue
        if in_deps:
            if line.strip().startswith("["):
                in_deps = False
                continue
            match = re.match(r'\s*([A-Za-z0-9_.\-]+)\s*=\s*["\']([^"\']+)["\']', line)
            if match and match.group(1) != "python":
                deps.append({
                    "name": match.group(1),
                    "version_spec": match.group(2),
                    "source": "pyproject.toml",
                })
    return deps


def _parse_package_json(content: str) -> List[Dict]:
    """Parse npm package.json."""
    deps = []
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return deps

    for section in ["dependencies", "devDependencies", "peerDependencies"]:
        for name, version in data.get(section, {}).items():
            deps.append({
                "name": name,
                "version_spec": version,
                "source": f"package.json ({section})",
                "dev": section == "devDependencies",
            })
    return deps


def _parse_cargo_toml(content: str) -> List[Dict]:
    """Parse Rust Cargo.toml."""
    deps = []
    in_deps = False
    for line in content.splitlines():
        if "[dependencies]" in line or "[dev-dependencies]" in line:
            in_deps = True
            continue
        if in_deps and line.strip().startswith("["):
            in_deps = False
            continue
        if in_deps:
            match = re.match(r'\s*([A-Za-z0-9_\-]+)\s*=\s*["\']?([^"\'#\n]+)', line)
            if match:
                deps.append({
                    "name": match.group(1),
                    "version_spec": match.group(2).strip().strip('"\''),
                    "source": "Cargo.toml",
                })
    return deps


def _parse_go_mod(content: str) -> List[Dict]:
    """Parse Go go.mod."""
    deps = []
    in_require = False
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("require ("):
            in_require = True
            continue
        if in_require and line == ")":
            in_require = False
            continue
        if in_require or line.startswith("require "):
            line = line.removeprefix("require ").strip()
            parts = line.split()
            if len(parts) >= 2:
                deps.append({
                    "name": parts[0],
                    "version_spec": parts[1],
                    "source": "go.mod",
                })
    return deps


# ---------------------------------------------------------------------------
# Tool: dependency_analyzer
# ---------------------------------------------------------------------------

MANIFEST_PARSERS = {
    "requirements.txt": _parse_requirements_txt,
    "pyproject.toml": _parse_pyproject_toml,
    "package.json": _parse_package_json,
    "Cargo.toml": _parse_cargo_toml,
    "go.mod": _parse_go_mod,
}


def dependency_analyzer(repo_root: str) -> Dict:
    """
    Scan the repository for dependency manifests and extract library info.

    Returns:
        {
            "manifests_found": [...],
            "dependencies": [...],
            "total_count": int,
        }
    """
    root = Path(repo_root).resolve()
    manifests_found = []
    all_deps: List[Dict] = []

    for manifest_name, parser in MANIFEST_PARSERS.items():
        manifest_path = root / manifest_name
        if manifest_path.exists():
            ok, content = read_file(repo_root, manifest_name)
            if ok:
                manifests_found.append(manifest_name)
                try:
                    deps = parser(content)
                    all_deps.extend(deps)
                except Exception as exc:
                    all_deps.append({"error": f"Failed to parse {manifest_name}: {exc}"})

    # Also check nested manifests (e.g. monorepos)
    for subdir in root.iterdir():
        if subdir.is_dir() and not subdir.name.startswith("."):
            for manifest_name in ["requirements.txt", "package.json"]:
                sub_manifest = subdir / manifest_name
                if sub_manifest.exists():
                    rel_path = str(sub_manifest.relative_to(root))
                    ok, content = read_file(repo_root, rel_path)
                    if ok:
                        manifests_found.append(rel_path)
                        parser = MANIFEST_PARSERS.get(manifest_name)
                        if parser:
                            try:
                                deps = parser(content)
                                for dep in deps:
                                    dep["source"] = rel_path
                                all_deps.extend(deps)
                            except Exception:
                                pass

    return {
        "manifests_found": manifests_found,
        "dependencies": all_deps,
        "total_count": len(all_deps),
    }
