"""
Markdown Parsing, Section Synonym Extraction, Template Placeholder Detection, and Sanitization Utilities
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

LIFECYCLE_STATUS_PATTERNS = [
    r"^\s*status:\s*(provisional|implementable|implemented|alpha|beta|ga|deprecated|withdrawn|rejected|deferred|replaced)\s*$",
    r"\bgraduation\s+criteria\b",
    r"\bimplementation\s+status\b",
    r"\balpha\s+timeline\b",
    r"\bbeta\s+timeline\b",
    r"\bga\s+timeline\b",
]


def sanitize_markdown(text: str) -> str:
    """
    Sanitizes raw markdown before section extraction:
    - Removes HTML comments (e.g. <!-- ... -->)
    - Removes template instruction comment blocks
    - Normalizes excessive whitespace while preserving paragraph breaks and code blocks
    - Preserves real prose and code blocks
    """
    if not text:
        return ""

    # 1. Remove HTML comments (single-line and multi-line)
    clean_text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # 2. Normalize 3+ consecutive newlines to 2 newlines
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)

    return clean_text.strip()


def has_template_placeholders(text: str) -> bool:
    """Check if text contains unresolved template placeholders like ${...}, {{...}}, {title of option...}."""
    if not text:
        return False
    return bool(TEMPLATE_PLACEHOLDER_REGEX.search(text))


def is_lifecycle_status_only(text: str) -> bool:
    """
    Returns True if text is merely lifecycle/graduation/status metadata
    (e.g., Alpha/Beta/GA timelines, status: implementable, graduation criteria only)
    without substantive architectural decision content.
    """
    if not text:
        return True

    clean_text = text.strip().lower()

    # Short status strings like 'Status: Implementable' or 'Status: Alpha'
    if len(clean_text) < 60:
        for pat in LIFECYCLE_STATUS_PATTERNS:
            if re.search(pat, clean_text, re.MULTILINE):
                return True
        if clean_text in ["implementable", "implemented", "provisional", "alpha", "beta", "ga"]:
            return True

    # Check if section consists almost entirely of status/graduation checklist items
    lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
    if lines:
        status_line_count = sum(
            1
            for line in lines
            if any(re.search(pat, line) for pat in LIFECYCLE_STATUS_PATTERNS)
            or line.startswith(("- [ ] alpha", "- [ ] beta", "- [ ] ga", "- [x] alpha", "- [x] beta", "- [x] ga"))
        )
        if (status_line_count / len(lines)) >= 0.60:
            return True

    return False


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
    Extract full markdown section content matching any heading synonym up to the next heading whose level <= N (where N is the matched heading level).
    If a section starts at heading level N, extraction stops at the next heading whose level is <= N.
    """
    for synonym in synonyms:
        # Match heading line capturing level hashes (1-6)
        pattern = rf"^\s*(#{{1,6}})\s+.*?\b{re.escape(synonym)}\b.*$"
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)

        if match:
            heading_hashes = match.group(1)
            heading_level = len(heading_hashes)
            start = match.end()
            remainder = text[start:]

            # Next heading pattern: next line starting with 1 to heading_level '#' characters
            next_heading_pattern = rf"\n#{{1,{heading_level}}}\s+"
            next_match = re.search(next_heading_pattern, remainder)
            if next_match:
                section_text = remainder[: next_match.start()]
            else:
                section_text = remainder

            content = section_text.strip()
            if content and len(content) >= 15:
                return content
    return ""


def extract_structured_items(text: str) -> list[str]:
    """
    Parses structured Markdown items (bullets, numbered lists, subheadings).
    Avoids converting ordinary prose paragraphs into separate individual items.
    """
    if not text:
        return []

    lines = text.splitlines()
    items: list[str] = []
    current_item: list[str] = []

    bullet_or_number_re = re.compile(r"^\s*([-*+]|\d+\.)\s+(.+)$")
    subheading_re = re.compile(r"^\s*#{2,6}\s+(.+)$")

    has_explicit_list_markers = any(bullet_or_number_re.match(l) for l in lines)

    if has_explicit_list_markers:
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            match = bullet_or_number_re.match(line)
            if match:
                if current_item:
                    items.append(" ".join(current_item).strip())
                    current_item = []
                current_item.append(match.group(2).strip())
            else:
                sub_match = subheading_re.match(line)
                if sub_match:
                    if current_item:
                        items.append(" ".join(current_item).strip())
                        current_item = []
                    items.append(sub_match.group(1).strip())
                elif current_item:
                    current_item.append(line_str)
        if current_item:
            items.append(" ".join(current_item).strip())
    else:
        # Check if section consists of subheadings (e.g. ### Option 1 ... ### Option 2)
        subheadings = [line for line in lines if subheading_re.match(line)]
        if subheadings and len(subheadings) >= 1:
            for line in lines:
                sub_match = subheading_re.match(line)
                if sub_match:
                    if current_item:
                        items.append("\n".join(current_item).strip())
                        current_item = []
                    items.append(sub_match.group(1).strip())
                elif line.strip() and current_item:
                    current_item.append(line.strip())
            if current_item:
                items.append("\n".join(current_item).strip())
        else:
            clean_text = text.strip()
            if clean_text:
                items.append(clean_text)

    return [item.strip("- *") for item in items if item and len(item.strip("- *")) > 0]
