"""
Repository Analyzer Agent — CodePilot AI.

Uses deterministic tools to produce a RepositorySummary without relying
on LLM for file discovery. The LLM is used only to generate a narrative
summary of the structural findings.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from config import MAX_FILES_PER_AGENT, get_llm
from graph.state import AnalysisState, AgentTraceEntry
from models.schemas import RepositorySummary
from tools.dependencies import dependency_analyzer
from tools.repository import (
    find_files,
    get_repo_metadata,
    read_file,
    repository_tree,
)

logger = logging.getLogger(__name__)

# Language detection by extension
LANGUAGE_MAP = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".jsx": "JavaScript (React)",
    ".tsx": "TypeScript (React)",
    ".java": "Java",
    ".go": "Go",
    ".rb": "Ruby",
    ".php": "PHP",
    ".cs": "C#",
    ".cpp": "C++",
    ".c": "C",
    ".rs": "Rust",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".scala": "Scala",
    ".sh": "Shell",
    ".sql": "SQL",
    ".html": "HTML",
    ".css": "CSS",
    ".vue": "Vue.js",
    ".svelte": "Svelte",
}

# Framework detection by file/pattern
FRAMEWORK_INDICATORS = {
    "Django": ["manage.py", "django.conf", "INSTALLED_APPS"],
    "Flask": ["from flask import", "Flask(__name__)", "flask"],
    "FastAPI": ["from fastapi import", "FastAPI()", "fastapi"],
    "Express.js": ["express()", "require('express')", "from 'express'"],
    "React": ["import React", "from 'react'", "ReactDOM"],
    "Vue.js": ["Vue.createApp", "from 'vue'", ".vue"],
    "Angular": ["@NgModule", "@Component", "angular.json"],
    "Spring": ["@SpringBootApplication", "spring-boot", "pom.xml"],
    "Rails": ["Rails.application", "Gemfile", "rails"],
    "Laravel": ["Illuminate\\", "artisan", "composer.json"],
    "Next.js": ["next.config.js", "next.config.ts", "from 'next'"],
    "pytest": ["import pytest", "def test_"],
    "Jest": ["jest.config.js", "describe(", "it("],
    "SQLAlchemy": ["from sqlalchemy", "SQLAlchemy"],
    "Celery": ["from celery import", "Celery("],
    "Docker": ["Dockerfile", "docker-compose.yml"],
    "Kubernetes": [".yaml", "apiVersion:", "kind: Deployment"],
}


def _detect_frameworks(repo_root: str, metadata: Dict) -> List[str]:
    """Detect frameworks by scanning key files for indicator patterns."""
    from tools.code_search import search_code

    detected = set()

    # Check file-based indicators
    all_file_names = {
        f.split("/")[-1] for f in find_files(repo_root)
    }
    for framework, indicators in FRAMEWORK_INDICATORS.items():
        for ind in indicators:
            if ind in all_file_names:
                detected.add(framework)
                break

    # Check content-based indicators for common frameworks
    for framework, indicators in FRAMEWORK_INDICATORS.items():
        if framework in detected:
            continue
        for ind in indicators:
            if ind.startswith("."):
                continue  # extension, not content pattern
            results = search_code(repo_root, ind, max_results=1)
            if results and not results[0].get("error"):
                detected.add(framework)
                break

    return sorted(detected)


def _detect_entry_points(repo_root: str) -> List[str]:
    """Find likely application entry points."""
    candidates = [
        "main.py", "app.py", "server.py", "run.py", "wsgi.py", "asgi.py",
        "manage.py", "index.js", "server.js", "app.js", "main.js",
        "main.go", "main.rs", "Main.java", "Program.cs",
    ]
    found = []
    for candidate in candidates:
        files = find_files(repo_root, name_pattern=candidate)
        found.extend(files)
    return found[:10]


def run_repository_analyzer(state: AnalysisState) -> Dict:
    """
    LangGraph node: Repository Analyzer.

    Analyzes the repository structure deterministically, then uses
    the LLM to generate a narrative summary.

    Args:
        state: Current graph state.

    Returns:
        State updates dict.
    """
    repo_root = state["repo_path"]
    logger.info(f"[RepositoryAnalyzer] Analyzing: {repo_root}")

    try:
        # --- Deterministic Analysis ---
        metadata = get_repo_metadata(repo_root)
        tree = repository_tree(repo_root)
        dep_info = dependency_analyzer(repo_root)
        entry_points = _detect_entry_points(repo_root)

        # Language detection
        ext_counts = metadata["extension_counts"]
        languages = sorted(
            [LANGUAGE_MAP[ext] for ext in ext_counts if ext in LANGUAGE_MAP],
            key=lambda lang: ext_counts.get(
                [k for k, v in LANGUAGE_MAP.items() if v == lang][0], 0
            ),
            reverse=True,
        )[:5]

        # Framework detection
        frameworks = _detect_frameworks(repo_root, metadata)

        # Count test files
        test_files = find_files(repo_root, name_pattern=None)
        test_file_count = sum(
            1 for f in test_files
            if "test" in f.lower() or "spec" in f.lower()
        )
        source_file_count = len(test_files) - test_file_count

        # Read README for structural insight
        readme_content = ""
        for rd in metadata["readme_files"]:
            ok, content = read_file(repo_root, rd)
            if ok:
                readme_content = content[:3000]  # first 3000 chars
                break

        # --- LLM Summary ---
        llm = get_llm()
        prompt = f"""You are a senior software engineer analyzing a code repository.

Repository Tree:
{tree[:3000]}

Languages detected: {', '.join(languages) or 'Unknown'}
Frameworks detected: {', '.join(frameworks) or 'None detected'}
Total source files: {metadata['total_files']}
Dependencies manifests: {', '.join(dep_info['manifests_found']) or 'None'}
Total dependencies: {dep_info['total_count']}
Entry points: {', '.join(entry_points) or 'None found'}
Test directories: {', '.join(metadata['test_directories']) or 'None'}
Config files: {', '.join(metadata['config_files'])}
Has CI/CD: {metadata['has_ci']}

README excerpt:
{readme_content[:1500] if readme_content else 'No README found.'}

Provide 5-7 key structural observations about this repository as a bulleted list.
Focus on: project type, architecture style, notable patterns, potential concerns.
Each bullet should be a single sentence. Be specific, not generic.
"""
        response = llm.invoke(prompt)
        key_findings_text = response.content

        key_findings = [
            line.strip().lstrip("•-* ").strip()
            for line in key_findings_text.splitlines()
            if line.strip() and (line.strip().startswith(("•", "-", "*")) or len(line.strip()) > 20)
        ][:7]

        repo_summary = RepositorySummary(
            root_path=repo_root,
            languages=languages or ["Unknown"],
            frameworks=frameworks,
            total_files=metadata["total_files"],
            source_files=source_file_count,
            test_files=test_file_count,
            config_files=len(metadata["config_files"]),
            has_readme=bool(metadata["readme_files"]),
            has_tests=bool(metadata["test_directories"]) or test_file_count > 0,
            has_ci=metadata["has_ci"],
            entry_points=entry_points,
            dependencies={
                "manifests": dep_info["manifests_found"],
                "total": dep_info["total_count"],
                "list": dep_info["dependencies"][:30],
            },
            directory_tree=tree,
            key_findings=key_findings,
        )

        trace_entry: AgentTraceEntry = {
            "node": "Repository Analyzer",
            "iteration": 1,
            "action": f"Analyzed {metadata['total_files']} files across {len(languages)} languages",
            "result": f"Found {len(frameworks)} frameworks, {dep_info['total_count']} dependencies, {test_file_count} test files",
        }

        logger.info(f"[RepositoryAnalyzer] Complete: {len(languages)} languages, {len(frameworks)} frameworks")

        return {
            "repo_summary": repo_summary,
            "agent_trace": [trace_entry],
            "current_node": "planner",
        }

    except Exception as exc:
        logger.error(f"[RepositoryAnalyzer] Error: {exc}", exc_info=True)
        return {
            "errors": [f"Repository Analyzer error: {str(exc)}"],
            "agent_trace": [{
                "node": "Repository Analyzer",
                "iteration": 1,
                "action": "Analysis failed",
                "result": str(exc),
            }],
            "current_node": "planner",
        }
