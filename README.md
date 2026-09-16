# 🧑‍💻 CodePilot AI

**Multi-Agent Software Engineering Assistant**

> An end-to-end portfolio project demonstrating agentic AI orchestration using **LangGraph**. CodePilot AI analyzes software repositories using multiple specialized agents and produces a professional engineering review covering security, code quality, testing, architecture, and documentation.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution](#solution)
- [Why LangGraph?](#why-langgraph)
- [Architecture](#architecture)
- [LangGraph Workflow](#langgraph-workflow)
- [Agent Responsibilities](#agent-responsibilities)
- [Shared State](#shared-state)
- [Conditional Routing](#conditional-routing)
- [Critic / Revision Loop](#critic--revision-loop)
- [Tool Architecture](#tool-architecture)
- [Security Considerations](#security-considerations)
- [Streamlit UI](#streamlit-ui)
- [Setup](#setup)
- [Environment Variables](#environment-variables)
- [Running Locally](#running-locally)
- [Docker](#docker)
- [Example Prompts](#example-prompts)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)

---

## Problem Statement

Code review is inherently a **multi-dimensional, iterative process**. A thorough review needs to:

- Identify security vulnerabilities with supporting evidence
- Find bugs and code quality issues
- Assess test coverage gaps
- Evaluate architectural decisions
- Review documentation quality

No single prompt to an LLM can do this reliably. The issues span multiple concerns, require investigation of different file types, and need validation — just like a real engineering team.

---

## Solution

CodePilot AI models software analysis as a **stateful, multi-agent workflow**:

1. **Repository Analyzer** maps the codebase structure deterministically
2. **Planner** decides which specialist agents are needed and what to focus on
3. **Specialist Agents** (Security, Code Review, Testing, Architecture, Documentation) investigate their domains in parallel
4. **Findings Aggregator** consolidates and deduplicates all findings
5. **Critic** reviews the findings for quality, evidence, and completeness
6. If findings are insufficient, the **Critic rejects them** and routes specific agents back for re-investigation
7. Once approved, the **Reporter** generates a professional engineering report

---

## Why LangGraph?

Software analysis is a **non-linear, iterative workflow** — not a simple chain.

### The Problems with Simple Chains

```
User → LLM → Answer
```

A simple chain or single prompt cannot:
- Run multiple specialized agents and merge their findings
- Validate findings with a critic and loop back for more evidence
- Protect against infinite loops (max iteration guard)
- Maintain shared state across multiple agents
- Route conditionally based on review outcomes

### What LangGraph Provides

LangGraph implements analysis as a **directed graph with typed state**:

```python
workflow = StateGraph(AnalysisState)
workflow.add_conditional_edges("critic", critic_routing, {
    "reporter": "reporter",        # Approved → generate report
    "re_investigate": "re_investigate",  # Rejected → re-investigate
})
```

Key capabilities used:
- **`StateGraph`** — shared state flows through all nodes
- **`add_conditional_edges`** — critic dynamically routes to reporter or re-investigation
- **Annotated reducers** — parallel agents merge findings without overwriting each other
- **Recursion limit** — prevents infinite loops (separate from application-level max iterations)

This makes the agentic behavior **explicit, inspectable, and controllable**.

---

## Architecture

```
codepilot-ai/
├── app.py                    # Streamlit UI
├── config.py                 # LLM configuration
├── requirements.txt
├── Dockerfile
│
├── graph/
│   ├── state.py              # TypedDict AnalysisState
│   └── workflow.py           # LangGraph StateGraph
│
├── agents/
│   ├── base.py               # Shared utilities
│   ├── repository_analyzer.py
│   ├── planner.py
│   ├── code_review.py
│   ├── security.py
│   ├── testing.py
│   ├── architecture.py
│   ├── documentation.py
│   ├── aggregator.py
│   ├── critic.py
│   └── reporter.py
│
├── tools/
│   ├── repository.py         # File I/O, ZIP extraction, tree
│   ├── code_search.py        # Pattern search
│   ├── dependencies.py       # Manifest parsing
│   └── secret_scanner.py     # Pattern-based secret detection
│
├── models/
│   └── schemas.py            # Pydantic models
│
└── tests/
    ├── test_repository.py
    ├── test_tools.py
    ├── test_schemas.py
    └── test_graph.py
```

---

## LangGraph Workflow

```
                     ZIP Upload
                         │
                         ▼
                REPOSITORY ANALYZER
            (deterministic tool analysis)
                         │
                         ▼
                      PLANNER
                (creates investigation plan)
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     CODE REVIEW    SECURITY       TESTING
       AGENT          AGENT         AGENT
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                   ARCHITECTURE AGENT
                         │
                         ▼
                   DOCUMENTATION AGENT
                         │
                         ▼
                FINDINGS AGGREGATOR
              (deduplicate + sort by severity)
                         │
                         ▼
                 ┌─────CRITIC──────┐
                 │  Reviews quality │
                 │  of all findings │
                 └─────────────────┘
                    │          │
                REJECT       APPROVE
                   │          │
                   ▼          ▼
           RE-INVESTIGATE   REPORTER
           (specific agents  (generates
            re-run with       final report)
            critic feedback)
                   │
                   └──→ FINDINGS AGGREGATOR
                               │
                               ▼
                           CRITIC
                     (max 3 revision cycles)
```

---

## Agent Responsibilities

| Agent | Input | Output |
|-------|-------|--------|
| **Repository Analyzer** | Repo path | `RepositorySummary` with languages, frameworks, structure |
| **Planner** | Repo summary + user request | `InvestigationPlan` with focus areas and agent selection |
| **Code Review** | Source files | Bugs, code smells, error handling issues |
| **Security** | Source files + secret scanner | Security vulnerabilities with evidence |
| **Testing** | Test and source files | Coverage gaps, missing edge cases |
| **Architecture** | Structure + key files | Coupling, SRP violations, design issues |
| **Documentation** | README + source files | Missing docs, docstrings, setup instructions |
| **Findings Aggregator** | All findings | Deduplicated, severity-sorted findings |
| **Critic** | All findings | `CriticReview` with approve/reject + reanalysis requests |
| **Reporter** | Approved findings | Scored `FinalReport` + full Markdown |

---

## Shared State

All agents share a single `AnalysisState` TypedDict:

```python
class AnalysisState(TypedDict):
    repo_path: str
    user_request: str
    repo_summary: Optional[RepositorySummary]
    investigation_plan: Optional[InvestigationPlan]
    all_findings: Annotated[List[Finding], _merge_findings]  # Reducer merges findings
    critic_review: Optional[CriticReview]
    revision_count: int         # How many critic→re-investigate cycles
    reanalysis_requests: List[ReanalysisRequest]
    final_report: Optional[FinalReport]
    agent_trace: Annotated[List[AgentTraceEntry], operator.add]
    errors: Annotated[List[str], operator.add]
```

The `Annotated` reducers are key — they allow **parallel agents to safely append findings** to the shared list without one agent overwriting another's work.

---

## Conditional Routing

The Critic's output determines the next step:

```python
def critic_routing(state: AnalysisState) -> str:
    review = state.get("critic_review")
    revision_count = state.get("revision_count", 0)

    if revision_count >= MAX_REVISION_CYCLES:
        return "reporter"  # Force exit after max cycles

    return "reporter" if review.approved else "re_investigate"
```

```python
workflow.add_conditional_edges(
    "critic",
    critic_routing,
    {
        "reporter": "reporter",
        "re_investigate": "re_investigate",
    },
)
```

---

## Critic / Revision Loop

The Critic is what makes this genuinely agentic.

**Critic checks each finding for:**
1. Sufficient code evidence (file + line reference)
2. Validity (not a false positive)
3. Appropriate severity level
4. Actionable recommendation
5. Duplicate detection
6. Missing investigation areas

**If rejected**, the Critic returns a structured `CriticReview`:

```json
{
  "approved": false,
  "overall_quality": 45,
  "invalid_findings": ["SQL Injection claim lacks code evidence"],
  "required_reanalysis": [
    {
      "agent_name": "security",
      "reason": "SQL injection claim needs code evidence",
      "focus_areas": ["database queries"],
      "specific_files": ["src/api/users.py"]
    }
  ],
  "comments": ["Security agent should examine the ORM layer"]
}
```

The Security Agent then **re-analyzes the specific files** and returns to the Critic.

**Maximum iterations:** 3 cycles (configurable via `MAX_REVISION_CYCLES`).

---

## Tool Architecture

All tools are **pure Python, read-only, deterministic** — no LLM, no subprocess, no code execution.

| Tool | Description |
|------|-------------|
| `repository_tree()` | Safe directory tree (excludes `__pycache__`, `node_modules`) |
| `read_file(path)` | Path-traversal-safe file reader |
| `find_files(ext)` | Find files by extension or name |
| `search_code(query)` | Regex/literal pattern search with context |
| `dependency_analyzer()` | Parses requirements.txt, package.json, Cargo.toml, go.mod |
| `secret_scanner()` | Pattern-based hardcoded secret detection |

Agents use tools **before** calling the LLM, grounding the analysis in real code.

---

## Security Considerations

> **⚠️ The uploaded repository is treated as untrusted input.**

The application:

- ✅ Uses `safe_extract_zip()` that checks for absolute paths and `..` traversal
- ✅ Uses `read_file()` that resolves paths and checks they stay within `repo_root`
- ✅ Never executes uploaded code, scripts, or binaries
- ✅ Never calls `subprocess`, `os.system`, `eval`, or `exec` on repository content
- ✅ Caps file sizes to prevent memory exhaustion
- ✅ Limits the number of files per agent pass
- ✅ Never logs API keys or secrets
- ✅ `.env` is in `.gitignore`

---

## Streamlit UI

The UI provides:

1. **Upload Screen** — ZIP upload, optional focus text, multi-select focus areas
2. **Live Progress** — Agent workflow progress shown in real-time
3. **Dashboard** — 6 health score cards (Overall Health, Security, Code Quality, Testing, Architecture, Documentation)
4. **Finding Cards** — Color-coded by severity with evidence, confidence, recommendations
5. **Tab View** — Separate tabs for Overview, Security, Code Quality, Testing, Architecture, Documentation, Recommendations
6. **Agent Trace** — Full execution trace showing the LangGraph workflow, iterations, and critic feedback
7. **Report Download** — Full Markdown report as a downloadable file

---

## Setup

### Prerequisites

- Python 3.11+
- A free [Groq API key](https://console.groq.com)

### Install

```bash
# Clone the repository
git clone https://github.com/yourname/codepilot-ai
cd codepilot-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | ✅ Yes | — | Groq API key from console.groq.com |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | Model to use |
| `REPO_TEMP_DIR` | No | `/tmp/codepilot_repos` | Temp dir for ZIP extraction |

---

## Running Locally

```bash
# Run the Streamlit app
streamlit run app.py

# Or with specific settings
streamlit run app.py --server.port=8501 --server.address=localhost

# Run tests (no API key needed)
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --tb=short
```

---

## Docker

```bash
# Build
docker build -t codepilot-ai .

# Run (pass your API key)
docker run -p 8501:8501 \
  -e GROQ_API_KEY=your_key_here \
  -e GROQ_MODEL=llama-3.3-70b-versatile \
  codepilot-ai

# Access at http://localhost:8501
```

---

## Example Prompts

### 1. Flask Web Application

**Repository:** A Flask REST API with user authentication and SQLite database.

**Prompt:**
```
Focus primarily on security vulnerabilities and authentication issues.
I'm particularly concerned about SQL injection and hardcoded credentials.
```

**Expected findings:**
- Hardcoded database credentials
- Parameterized query issues
- Missing rate limiting
- JWT configuration weaknesses

---

### 2. Node.js Microservice

**Repository:** An Express.js microservice with JWT authentication.

**Prompt:**
```
Analyze for security, testing gaps, and architectural concerns.
The service handles sensitive user data.
```

**Expected findings:**
- Missing input validation
- Insufficient test coverage
- No integration tests
- Missing error handling middleware

---

### 3. Python Data Pipeline

**Repository:** A data processing pipeline with Pandas and database integration.

**Prompt:**
```
Focus on code quality, performance issues, and testing.
This runs in production on large datasets.
```

**Expected findings:**
- N+1 query patterns
- Memory inefficiency with large DataFrames
- Missing data validation
- Insufficient unit tests for edge cases

---

## Limitations

- **Not a professional security scanner.** This is a portfolio demonstration. Do not use as a substitute for dedicated SAST tools (Snyk, Semgrep, CodeQL), DAST tools, or professional penetration testing.
- **LLM hallucination.** Language models can generate plausible-sounding but incorrect findings. All findings require manual verification.
- **Context window limits.** Very large codebases are analyzed by sampling files. The tool caps file reads per agent to avoid token limits.
- **Static analysis only.** Dynamic issues (race conditions, real runtime behavior) cannot be detected by static analysis alone.
- **Language coverage.** Deterministic tools work best for Python. Other languages receive LLM analysis but less deterministic scanning.
- **No real CVE database.** Dependency vulnerability checking does not query NVD or similar databases.

---

## Future Enhancements

- [ ] **GitHub/GitLab integration** — Analyze repositories directly from URL
- [ ] **Incremental analysis** — Only re-analyze changed files (git diff integration)
- [ ] **CVE database lookup** — Check dependencies against NVD/OSV
- [ ] **Multi-model support** — OpenAI, Anthropic, Ollama (local models)
- [ ] **Custom agent plugins** — Let users add domain-specific agents
- [ ] **Historical comparison** — Track health score over time
- [ ] **GitHub Actions integration** — Run as a CI/CD check
- [ ] **Export formats** — SARIF, PDF, JIRA tickets
- [ ] **Language-specific depth** — Deeper AST-based analysis per language
- [ ] **Streaming UI** — Real-time finding display as agents run

---

## Interview Explanation: LangGraph Architecture

> **"Why did you use LangGraph instead of a simple chain?"**

A simple LLM chain — `prompt → response` — can't model iterative, multi-stakeholder workflows. Code review is inherently:

1. **Multi-dimensional:** Security requires different expertise than testing or architecture
2. **Iterative:** An initial analysis may lack evidence; a critic should request more investigation
3. **Conditional:** The workflow branches based on whether findings are approved or rejected
4. **Bounded:** Without iteration limits, re-investigation loops could run forever

LangGraph models this as a directed graph with typed shared state. Each specialist agent is a node that reads the shared `AnalysisState` and writes its findings back. The `add_conditional_edges` API routes the workflow dynamically based on the Critic's decision — approve → report, reject → re-investigate.

The `Annotated` reducer on `all_findings` solves a real concurrency problem: three agents (Code Review, Security, Testing) run "in parallel" in the graph. Without a reducer, they'd overwrite each other's findings. With `_merge_findings` as a reducer, each agent safely appends its findings to the shared list.

This is a portfolio project, but it demonstrates exactly the patterns used in production agentic systems: typed state, conditional routing, revision loops, and maximum iteration guards.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph |
| LLM | Groq + Llama 3.3 70B |
| LLM SDK | LangChain + langchain-groq |
| UI | Streamlit |
| Schemas | Pydantic v2 |
| Testing | pytest |
| Containerization | Docker |
| Language | Python 3.11+ |

---

*Built as a portfolio project demonstrating advanced agentic AI patterns.*
*Not intended for production security use without professional review.*