"""
Markdown Parsing, Section Synonym Extraction, and Template Placeholder Detection Utilities
"""

import re
from pathlib import Path

TEMPLATE_PLACEHOLDER_REGEX = re.compile(
    r"(\$\{[A-Za-z0-9_\-]+\}|"
    r"\{\{[A-Za-z0-9_\-\s]+\}\}|"
    r"\{[a-z0-9_\-\s]*(option|title|problem|decision|author|date|status)[a-z0-9_\-\s]*\}|"
    r"\[[a-z0-9_\-\s]*(short title|title of|name of option|solved problem|YYYY-MM-DD|status)[a-z0-9_\-\s]*\])",
    re.IGNORECASE,
)

BOILERPLATE_FILE_PATTERNS = [
    "changelog",
    "release",
    "version",
    "migration",
    "adr-template",
    "kep-template",
    "0000-kep-process",
    "odh-adr-0000-template",
    "adr000-template",
    "template.md",
    "contributing.md",
    "code_of_conduct.md",
    "authors.md",
    "license.md",
    "examples.md",
    "tooling.md",
    "skill.md",
    "adr-guide.md",
]

CONTEXT_SYNONYMS = [
    "context and problem statement",
    "context",
    "motivation",
    "problem statement",
    "problem",
    "background",
    "summary",
    "rationale",
    "goals",
    "user stories",
]

DECISION_SYNONYMS = [
    "decision outcome",
    "decision",
    "proposal",
    "design details",
    "design",
    "solution",
    "recommendation",
    "outcome",
    "chosen option",
    "approach",
    "implementation",
    "architecture",
]

CONSEQUENCE_SYNONYMS = [
    "consequences",
    "positive consequences",
    "negative consequences",
    "risks and mitigations",
    "risks",
    "tradeoffs",
    "trade-offs",
    "pros and cons",
]

ALTERNATIVE_SYNONYMS = [
    "considered options",
    "alternatives considered",
    "alternatives",
    "rejected options",
    "options",
]


def has_template_placeholders(text: str) -> bool:
    """Check if text contains unresolved template placeholders like ${...}, {{...}}, {title of option...}."""
    if not text:
        return False
    return bool(TEMPLATE_PLACEHOLDER_REGEX.search(text))


def is_boilerplate_filename(file_path: Path | str, source_id: str = "") -> bool:
    """
    Source-aware & path-aware check if a file matches boilerplate/template/changelog documentation.
    Whitelists keps/<sig>/<kep-id>/README.md as genuine Kubernetes KEP documents.
    """
    fpath = Path(file_path)
    fname_lower = fpath.name.lower()
    path_str_lower = str(fpath).lower().replace("\\", "/")

    # KEP Whitelist: keps/<sig>/<kep-id>/README.md is a genuine KEP!
    if (
        source_id == "k8s_keps"
        and fname_lower == "readme.md"
        and "0000-kep-process" not in path_str_lower
        and "nnnn-kep-template" not in path_str_lower
    ):
        parts = fpath.parts
        if any(p == "keps" for p in parts[:-1]):
            return False

    # Backstage / ODH Whitelist: adrs/<id>/README.md or architecture-decisions/README.md inside decision folders
    if (
        fname_lower == "readme.md"
        and ("architecture-decisions" in path_str_lower or "adrs" in path_str_lower)
        and "template" not in path_str_lower
        and "process" not in path_str_lower
    ):
        parts = fpath.parts
        if len(parts) >= 3:
            return False

    for pattern in BOILERPLATE_FILE_PATTERNS:
        if pattern in fname_lower or (pattern in path_str_lower and pattern != "readme.md"):
            return True

    # Generic root README.md
    return fname_lower == "readme.md"


def extract_markdown_section(text: str, synonyms: list[str]) -> str:
    """
    Extract full markdown section content matching any heading synonym up to the next level-1 or level-2 heading.
    """
    for synonym in synonyms:
        # Exact heading match, e.g. '## Decision' or '### Proposal'
        pattern = rf"^\s*#{{1,3}}\s+{re.escape(synonym)}\s*$"
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if not match:
            # Looser match, e.g. '## 3. Decision Outcome' or '### Proposal Summary'
            pattern = rf"^\s*#{{1,3}}\s+.*?\b{re.escape(synonym)}\b.*$"
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)

        if match:
            start = match.end()
            remainder = text[start:]
            # Match next level-1 or level-2 heading at line start
            next_match = re.search(r"\n#{1,2}\s+", remainder)
            if next_match:
                section_text = remainder[: next_match.start()]
            else:
                section_text = remainder
            content = section_text.strip()
            if content and len(content) >= 15:
                return content
    return ""
