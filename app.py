"""
CodePilot AI — Multi-Agent Software Engineering Assistant
Streamlit Application Entry Point

This is the main UI for CodePilot AI.
It accepts a ZIP file containing a repository, runs the LangGraph
multi-agent analysis workflow, and displays a professional engineering report.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Page config must be the first Streamlit call
st.set_page_config(
    page_title="CodePilot AI — Engineering Assistant",
    page_icon="🧑‍💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Headers */
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }

    /* Hero title */
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: #a0aec0;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }

    /* Score cards */
    .score-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .score-card:hover {
        background: rgba(255,255,255,0.08);
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateY(-2px);
    }

    .score-value {
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.5rem 0;
    }

    .score-label {
        font-size: 0.85rem;
        color: #a0aec0;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Finding cards */
    .finding-card {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid;
        transition: all 0.2s ease;
    }

    .finding-card:hover {
        background: rgba(255,255,255,0.07);
        transform: translateX(2px);
    }

    .finding-critical { border-left-color: #fc5c7d; }
    .finding-high     { border-left-color: #f6a623; }
    .finding-medium   { border-left-color: #f7ce68; }
    .finding-low      { border-left-color: #6dd5fa; }
    .finding-info     { border-left-color: #a8edea; }

    /* Severity badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .badge-critical { background: rgba(252,92,125,0.2); color: #fc5c7d; }
    .badge-high     { background: rgba(246,166,35,0.2); color: #f6a623; }
    .badge-medium   { background: rgba(247,206,104,0.2); color: #f7ce68; }
    .badge-low      { background: rgba(109,213,250,0.2); color: #6dd5fa; }
    .badge-info     { background: rgba(168,237,234,0.2); color: #a8edea; }

    /* Progress step */
    .step-done   { color: #68d391; }
    .step-active { color: #f6e05e; }
    .step-error  { color: #fc8181; }
    .step-pending{ color: #718096; }

    /* Upload area */
    .upload-container {
        background: rgba(255,255,255,0.04);
        border: 2px dashed rgba(102,126,234,0.4);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .upload-container:hover {
        border-color: rgba(102,126,234,0.8);
        background: rgba(102,126,234,0.06);
    }

    /* Code blocks */
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
    }

    /* Agent trace */
    .trace-entry {
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid rgba(102,126,234,0.5);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #a0aec0;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(102,126,234,0.3) !important;
        color: #fff !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15,15,26,0.95);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s ease;
        cursor: pointer;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.4);
    }

    /* Metric */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* Alert boxes */
    .alert-warning {
        background: rgba(246,166,35,0.1);
        border: 1px solid rgba(246,166,35,0.3);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #f6a623;
        margin: 0.5rem 0;
    }

    .alert-success {
        background: rgba(104,211,145,0.1);
        border: 1px solid rgba(104,211,145,0.3);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #68d391;
        margin: 0.5rem 0;
    }

    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        margin: 1.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: Score Color
# ---------------------------------------------------------------------------

def score_color(score: int) -> str:
    if score >= 80:
        return "#68d391"  # green
    elif score >= 60:
        return "#f6e05e"  # yellow
    elif score >= 40:
        return "#f6a623"  # orange
    else:
        return "#fc5c7d"  # red


def score_emoji(score: int) -> str:
    if score >= 80:
        return "🟢"
    elif score >= 60:
        return "🟡"
    elif score >= 40:
        return "🟠"
    else:
        return "🔴"


def severity_badge(severity: str) -> str:
    classes = {
        "Critical": "badge badge-critical",
        "High": "badge badge-high",
        "Medium": "badge badge-medium",
        "Low": "badge badge-low",
        "Info": "badge badge-info",
    }
    return f'<span class="{classes.get(severity, "badge")}">{severity}</span>'


def finding_card_class(severity: str) -> str:
    return {
        "Critical": "finding-card finding-critical",
        "High": "finding-card finding-high",
        "Medium": "finding-card finding-medium",
        "Low": "finding-card finding-low",
        "Info": "finding-card finding-info",
    }.get(severity, "finding-card")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0;">
            <div style="font-size:2.5rem;">🧑‍💻</div>
            <div style="font-size:1.1rem; font-weight:700; color:#667eea;">CodePilot AI</div>
            <div style="font-size:0.8rem; color:#718096;">Multi-Agent Engineering Assistant</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("### 🤖 Agent Pipeline")
        agents = [
            ("🔍", "Repository Analyzer"),
            ("📋", "Planning Agent"),
            ("💻", "Code Review Agent"),
            ("🔐", "Security Agent"),
            ("🧪", "Testing Agent"),
            ("🏗️", "Architecture Agent"),
            ("📖", "Documentation Agent"),
            ("📊", "Findings Aggregator"),
            ("⚖️", "Critic Agent"),
            ("📄", "Reporter"),
        ]
        for icon, name in agents:
            st.markdown(f"<div style='color:#a0aec0; font-size:0.85rem; padding:0.2rem 0;'>{icon} {name}</div>", unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("### ⚡ Powered By")
        st.markdown("""
        <div style="color:#718096; font-size:0.82rem; line-height:1.8;">
        🦾 <b>LangGraph</b> — Workflow<br>
        🤖 <b>Groq + Llama 3.3</b> — LLM<br>
        🐍 <b>Python</b> — Core<br>
        🎈 <b>Streamlit</b> — UI<br>
        📦 <b>Pydantic</b> — Schemas
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("### ⚠️ Disclaimer")
        st.markdown("""
        <div style="color:#718096; font-size:0.78rem; line-height:1.6;">
        This is a portfolio demonstration tool.
        AI-generated findings require manual verification.
        Not a replacement for professional security audits or SAST tools.
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Hero Section
# ---------------------------------------------------------------------------

def render_hero():
    st.markdown('<h1 class="hero-title">🧑‍💻 CodePilot AI</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Multi-Agent Software Engineering Assistant — Powered by LangGraph</p>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Upload Section
# ---------------------------------------------------------------------------

def render_upload_section():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="upload-container">
            <div style="font-size:2rem; margin-bottom:0.5rem;">📦</div>
            <div style="color:#a0aec0; font-size:0.9rem;">Upload your repository as a ZIP file</div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload Repository ZIP",
            type=["zip"],
            help="Upload a ZIP file containing your codebase. Max 200MB.",
            label_visibility="collapsed",
        )

        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

        user_request = st.text_area(
            "💬 What should I focus on? (optional)",
            placeholder="e.g., Focus primarily on security vulnerabilities and testing gaps. I'm concerned about the authentication module.",
            height=100,
            help="Describe any specific areas you want the agents to prioritize.",
        )

        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

        focus_options = ["Security", "Bugs", "Testing", "Code Quality", "Architecture", "Documentation", "Performance"]
        selected_focus = st.multiselect(
            "🎯 Focus Areas (optional)",
            focus_options,
            help="Select specific areas to emphasize in the analysis.",
        )

        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

        analyze_btn = st.button(
            "🚀 Analyze Repository",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None,
        )

    return uploaded_file, user_request, selected_focus, analyze_btn


# ---------------------------------------------------------------------------
# Workflow Progress Display
# ---------------------------------------------------------------------------

def render_workflow_progress(progress_container, completed_steps: List[str], current_step: str, trace=None):
    """Render real-time workflow progress."""
    workflow_steps = [
        ("repository_analyzer", "Repository Analyzer", "🔍"),
        ("planner", "Planning Agent", "📋"),
        ("code_review", "Code Review Agent", "💻"),
        ("security", "Security Agent", "🔐"),
        ("testing", "Testing Agent", "🧪"),
        ("architecture", "Architecture Agent", "🏗️"),
        ("documentation", "Documentation Agent", "📖"),
        ("findings_aggregator", "Findings Aggregator", "📊"),
        ("critic", "Critic Agent", "⚖️"),
        ("reporter", "Reporter", "📄"),
    ]

    with progress_container.container():
        st.markdown("### 🔄 Agent Workflow")

        for step_id, step_name, icon in workflow_steps:
            if step_id in completed_steps:
                st.markdown(
                    f'<div class="step-done">✓ {icon} {step_name}</div>',
                    unsafe_allow_html=True,
                )
            elif step_id == current_step:
                st.markdown(
                    f'<div class="step-active">⟳ {icon} {step_name} <span style="color:#718096; font-size:0.8em;">(running...)</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="step-pending">○ {icon} {step_name}</div>',
                    unsafe_allow_html=True,
                )

        # Show critic feedback if available
        if trace:
            critic_entries = [t for t in trace if t.get("node") == "Critic Agent"]
            for entry in critic_entries:
                if "Rejected" in entry.get("action", ""):
                    st.markdown(
                        f'<div class="alert-warning">⚠️ {entry["action"]}</div>',
                        unsafe_allow_html=True,
                    )
                elif "Approved" in entry.get("result", ""):
                    st.markdown(
                        f'<div class="alert-success">✓ Critic approved findings</div>',
                        unsafe_allow_html=True,
                    )


# ---------------------------------------------------------------------------
# Dashboard Rendering
# ---------------------------------------------------------------------------

def render_score_card(label: str, score: int, icon: str):
    color = score_color(score)
    st.markdown(f"""
    <div class="score-card">
        <div style="font-size:1.5rem;">{icon}</div>
        <div class="score-value" style="color:{color};">{score}</div>
        <div class="score-label">{label}</div>
        <div style="font-size:0.75rem; color:#718096;">/ 100</div>
    </div>
    """, unsafe_allow_html=True)


def render_finding_card(finding):
    severity = finding.severity.value
    card_class = finding_card_class(severity)
    badge_html = severity_badge(severity)

    evidence_html = ""
    if finding.evidence:
        evidence_html = f"""
        <details style="margin-top:0.8rem;">
            <summary style="cursor:pointer; color:#a0aec0; font-size:0.82rem;">📎 Evidence</summary>
            <pre style="background:rgba(0,0,0,0.3); border-radius:6px; padding:0.8rem; margin-top:0.5rem; font-size:0.8rem; overflow-x:auto; white-space:pre-wrap;">{finding.evidence[:400]}</pre>
        </details>
        """

    st.markdown(f"""
    <div class="{card_class}">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:1rem;">
            <div>
                {badge_html}
                <span style="font-weight:600; font-size:0.95rem; margin-left:0.5rem;">{finding.title}</span>
            </div>
            <div style="text-align:right; flex-shrink:0;">
                <div style="font-size:0.75rem; color:#718096;">Confidence</div>
                <div style="font-size:1rem; font-weight:700; color:{score_color(finding.confidence_pct())};">{finding.confidence_pct()}%</div>
            </div>
        </div>
        <div style="margin-top:0.5rem; color:#a0aec0; font-size:0.85rem;">
            {"📁 <code>" + finding.file + "</code>" if finding.file else ""}
            {"&nbsp;&nbsp;🔢 Line " + str(finding.line) if finding.line else ""}
            <span style="margin-left:0.5rem; color:#718096; font-size:0.8rem;">[{finding.source.value}]</span>
        </div>
        <div style="margin-top:0.6rem; color:#e2e8f0; font-size:0.88rem; line-height:1.5;">{finding.description[:300]}</div>
        {evidence_html}
        <div style="margin-top:0.8rem; background:rgba(102,126,234,0.1); border-radius:6px; padding:0.6rem 0.8rem; font-size:0.83rem;">
            <span style="color:#667eea; font-weight:600;">💡 Recommendation:</span>
            <span style="color:#cbd5e0;"> {finding.recommendation[:200]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_dashboard(final_state):
    """Render the complete analysis dashboard."""
    from models.schemas import FindingCategory, Severity

    report = final_state.get("final_report")
    findings = final_state.get("all_findings", [])
    repo_summary = final_state.get("repo_summary")
    trace = final_state.get("agent_trace", [])
    critic_review = final_state.get("critic_review")

    if not report:
        st.error("Analysis did not produce a final report.")
        return

    # ── Scores ──
    st.markdown("## 📊 Repository Health Dashboard")
    cols = st.columns(6)
    score_configs = [
        ("Overall Health", report.health_score, "🏥"),
        ("Security", report.security_risk_score, "🔐"),
        ("Code Quality", report.code_quality_score, "💻"),
        ("Testing", report.testing_score, "🧪"),
        ("Architecture", report.architecture_score, "🏗️"),
        ("Documentation", report.documentation_score, "📖"),
    ]
    for col, (label, score, icon) in zip(cols, score_configs):
        with col:
            render_score_card(label, score, icon)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Findings summary ──
    if findings:
        critical = sum(1 for f in findings if f.severity == Severity.CRITICAL)
        high = sum(1 for f in findings if f.severity == Severity.HIGH)
        medium = sum(1 for f in findings if f.severity == Severity.MEDIUM)
        low = sum(1 for f in findings if f.severity == Severity.LOW)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Findings", len(findings))
        m2.metric("🔴 Critical", critical)
        m3.metric("🟠 High", high)
        m4.metric("🟡 Medium", medium)
        m5.metric("🟢 Low", low)

    # ── Tabs ──
    tab_overview, tab_security, tab_code, tab_testing, tab_arch, tab_doc, tab_rec, tab_trace = st.tabs([
        "📋 Overview",
        "🔐 Security",
        "💻 Code Quality",
        "🧪 Testing",
        "🏗️ Architecture",
        "📖 Documentation",
        "🎯 Recommendations",
        "🔬 Agent Trace",
    ])

    # --- Overview Tab ---
    with tab_overview:
        st.markdown("### 📝 Executive Summary")
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.04); border-radius:12px; padding:1.5rem; line-height:1.8; color:#e2e8f0;">{report.executive_summary}</div>',
            unsafe_allow_html=True,
        )

        if repo_summary:
            st.markdown("### 🗂️ Repository Overview")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Languages:** {', '.join(repo_summary.languages)}")
                st.markdown(f"**Frameworks:** {', '.join(repo_summary.frameworks) or 'None detected'}")
                st.markdown(f"**Total Files:** {repo_summary.total_files}")
                st.markdown(f"**Source Files:** {repo_summary.source_files}")
                st.markdown(f"**Test Files:** {repo_summary.test_files}")
            with col2:
                st.markdown(f"**Has README:** {'✅' if repo_summary.has_readme else '❌'}")
                st.markdown(f"**Has Tests:** {'✅' if repo_summary.has_tests else '❌'}")
                st.markdown(f"**Has CI/CD:** {'✅' if repo_summary.has_ci else '❌'}")
                st.markdown(f"**Dependencies:** {repo_summary.dependencies.get('total', 0)}")
                entry_points = repo_summary.entry_points
                if entry_points:
                    st.markdown(f"**Entry Points:** {', '.join(entry_points[:3])}")

            if repo_summary.key_findings:
                st.markdown("### 🔑 Key Structural Observations")
                for obs in repo_summary.key_findings:
                    st.markdown(f"• {obs}")

            with st.expander("📂 Directory Tree"):
                st.code(repo_summary.directory_tree, language=None)

        if critic_review:
            st.markdown("### ⚖️ Analysis Quality")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Quality Score", f"{critic_review.overall_quality}/100")
                st.metric("Revision Cycles", final_state.get("revision_count", 0))
            with col2:
                if critic_review.comments:
                    for comment in critic_review.comments[:3]:
                        st.info(comment)

    # --- Security Tab ---
    with tab_security:
        sec_findings = [f for f in findings if f.category == FindingCategory.SECURITY]
        if sec_findings:
            st.warning(
                "⚠️ **Security Disclaimer:** These findings are generated by AI pattern matching and heuristics. "
                "Manual verification by a qualified security engineer is required. "
                "This tool does NOT replace professional security audits.",
                icon="⚠️",
            )
            st.markdown(f"**{len(sec_findings)} security findings found:**")
            for f in sec_findings:
                render_finding_card(f)
        else:
            st.success("✅ No security findings identified in this analysis.")

    # --- Code Quality Tab ---
    with tab_code:
        code_findings = [f for f in findings if f.category.value in ("Code Quality", "Bug")]
        if code_findings:
            st.markdown(f"**{len(code_findings)} code quality findings:**")
            for f in code_findings:
                render_finding_card(f)
        else:
            st.success("✅ No code quality issues identified.")

    # --- Testing Tab ---
    with tab_testing:
        test_findings = [f for f in findings if f.category == FindingCategory.TESTING]
        if repo_summary:
            st.markdown(f"**Test files:** {repo_summary.test_files} | **Source files:** {repo_summary.source_files}")
        if test_findings:
            st.markdown(f"**{len(test_findings)} testing gaps identified:**")
            for f in test_findings:
                render_finding_card(f)
        else:
            st.success("✅ Testing analysis found no significant gaps.")

    # --- Architecture Tab ---
    with tab_arch:
        arch_findings = [f for f in findings if f.category == FindingCategory.ARCHITECTURE]
        if arch_findings:
            st.markdown(f"**{len(arch_findings)} architectural findings:**")
            for f in arch_findings:
                render_finding_card(f)
        else:
            st.success("✅ No significant architectural issues identified.")

    # --- Documentation Tab ---
    with tab_doc:
        doc_findings = [f for f in findings if f.category == FindingCategory.DOCUMENTATION]
        if doc_findings:
            st.markdown(f"**{len(doc_findings)} documentation gaps:**")
            for f in doc_findings:
                render_finding_card(f)
        else:
            st.success("✅ Documentation appears adequate.")

    # --- Recommendations Tab ---
    with tab_rec:
        st.markdown("### 🎯 Prioritized Recommendations")
        st.markdown("_Ordered from most to least urgent:_")
        if report.recommended_actions:
            for i, action in enumerate(report.recommended_actions, 1):
                icon = "🔴" if i <= 2 else "🟠" if i <= 4 else "🟡" if i <= 7 else "🟢"
                st.markdown(
                    f'<div style="background:rgba(255,255,255,0.04); border-radius:8px; padding:0.8rem 1rem; margin-bottom:0.5rem;">'
                    f'<span style="font-weight:700; color:#667eea; margin-right:0.5rem;">{icon} {i}.</span>'
                    f'<span style="color:#e2e8f0;">{action}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Full report download
        if report.full_report_markdown:
            st.markdown("---")
            st.markdown("### 📄 Full Report")
            st.download_button(
                "⬇️ Download Full Report (Markdown)",
                data=report.full_report_markdown,
                file_name="codepilot_engineering_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

    # --- Agent Trace Tab ---
    with tab_trace:
        st.markdown("### 🔬 Agent Execution Trace")
        st.markdown(
            "_This trace shows the LangGraph workflow execution — each agent, iteration, and the critic feedback loop._"
        )
        if trace:
            for entry in trace:
                node = entry.get("node", "Unknown")
                iteration = entry.get("iteration", 1)
                action = entry.get("action", "")
                result = entry.get("result", "")

                is_critic = "Critic" in node
                is_reinvestigate = "Re" in node or "re_" in node

                bg_color = "rgba(102,126,234,0.08)"
                if is_critic and "Rejected" in action:
                    bg_color = "rgba(252,92,125,0.08)"
                elif is_critic and "Approved" in result:
                    bg_color = "rgba(104,211,145,0.08)"

                st.markdown(f"""
                <div class="trace-entry" style="background:{bg_color};">
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:#667eea; font-weight:600;">{node}</span>
                        <span style="color:#718096;">Iteration {iteration}</span>
                    </div>
                    <div style="color:#a0aec0; margin-top:0.2rem;">▸ {action}</div>
                    <div style="color:#68d391; margin-top:0.2rem; font-size:0.78rem;">↳ {result}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No trace entries available.")


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main():
    inject_css()
    render_sidebar()
    render_hero()

    # LLM Config check
    from config import check_llm_config
    llm_ok, llm_msg = check_llm_config()
    if not llm_ok:
        st.error(
            f"⚠️ **LLM Configuration Error:** {llm_msg}\n\n"
            "Please add your `GROQ_API_KEY` to the `.env` file:\n"
            "```\nGROQ_API_KEY=your_key_here\nGROQ_MODEL=llama-3.3-70b-versatile\n```\n"
            "Get your free API key at: https://console.groq.com",
            icon="🔑",
        )
        st.stop()

    # Check for existing results
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False
    if "final_state" not in st.session_state:
        st.session_state.final_state = None

    # Upload section
    if not st.session_state.analysis_complete:
        uploaded_file, user_request, selected_focus, analyze_btn = render_upload_section()

        if analyze_btn and uploaded_file is not None:
            _run_analysis(uploaded_file, user_request, selected_focus)

    # Display results
    if st.session_state.analysis_complete and st.session_state.final_state:
        st.success("✅ Analysis complete! Scroll down to view the full report.")

        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄 Analyze Another Repository"):
                st.session_state.analysis_complete = False
                st.session_state.final_state = None
                st.rerun()

        render_dashboard(st.session_state.final_state)


def _run_analysis(uploaded_file, user_request: str, selected_focus: List[str]):
    """Extract ZIP and run analysis with progress display."""
    from tools.repository import safe_extract_zip
    from graph.workflow import run_analysis

    # Create temp directory
    tmp_dir = tempfile.mkdtemp(prefix="codepilot_")

    try:
        # Save ZIP
        zip_path = os.path.join(tmp_dir, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.read())

        # Extract
        extract_dir = os.path.join(tmp_dir, "repo")
        success, repo_root, error = safe_extract_zip(zip_path, extract_dir)

        if not success:
            st.error(f"❌ ZIP extraction failed: {error}")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return

        # Setup progress display
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 🤖 Running Multi-Agent Analysis")
            progress_placeholder = st.empty()

            # Show initial workflow
            steps_done = []
            with progress_placeholder.container():
                st.markdown("**Starting analysis...**")

        with col2:
            stats_placeholder = st.empty()

        # Run the full analysis
        with st.spinner(""):
            start_time = time.time()

            # Progress indicator while running
            with progress_placeholder.container():
                st.markdown("**⟳ LangGraph workflow executing...**")
                steps = [
                    "🔍 Analyzing repository structure...",
                    "📋 Creating investigation plan...",
                    "💻 Running code review...",
                    "🔐 Running security analysis...",
                    "🧪 Analyzing tests...",
                    "🏗️ Reviewing architecture...",
                    "📖 Checking documentation...",
                    "📊 Aggregating findings...",
                    "⚖️ Critic reviewing findings...",
                    "📄 Generating final report...",
                ]
                for step in steps:
                    st.markdown(f"<div style='color:#a0aec0;'>{step}</div>", unsafe_allow_html=True)

            final_state = run_analysis(
                repo_path=repo_root,
                user_request=user_request,
                focus_areas=selected_focus,
            )

        elapsed = time.time() - start_time

        # Show completed workflow
        with progress_placeholder.container():
            trace = final_state.get("agent_trace", [])
            nodes_visited = list(dict.fromkeys(t.get("node", "") for t in trace))

            for node in nodes_visited:
                st.markdown(f"<div style='color:#68d391;'>✓ {node}</div>", unsafe_allow_html=True)

            # Critic feedback
            critic_entries = [t for t in trace if t.get("node") == "Critic Agent"]
            for entry in critic_entries:
                if "Rejected" in entry.get("action", ""):
                    st.markdown(
                        f"<div style='color:#f6a623;'>⚠️ {entry['action']}</div>",
                        unsafe_allow_html=True,
                    )
                elif "Approved" in entry.get("result", ""):
                    st.markdown(
                        "<div style='color:#68d391;'>✓ Critic approved findings</div>",
                        unsafe_allow_html=True,
                    )

        with stats_placeholder.container():
            findings = final_state.get("all_findings", [])
            report = final_state.get("final_report")
            st.metric("⏱️ Analysis Time", f"{elapsed:.0f}s")
            st.metric("📊 Total Findings", len(findings))
            if report:
                st.metric("🏥 Health Score", f"{report.health_score}/100")

        # Handle errors
        errors = final_state.get("errors", [])
        if errors:
            with st.expander(f"⚠️ {len(errors)} warnings during analysis"):
                for err in errors:
                    st.warning(err)

        st.session_state.final_state = final_state
        st.session_state.analysis_complete = True
        st.rerun()

    except Exception as exc:
        st.error(f"❌ Analysis failed: {exc}")
        logger.error(f"Analysis error: {exc}", exc_info=True)
    finally:
        # Cleanup temp files
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
