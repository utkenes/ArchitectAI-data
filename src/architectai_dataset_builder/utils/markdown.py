"""
Markdown Parsing and Template Placeholder Detection Utilities
"""

import re

TEMPLATE_PLACEHOLDER_REGEX = re.compile(
    r"(\$\{[A-Za-z0-9_\-]+\}|"
    r"\{\{[A-Za-z0-9_\-\s]+\}\}|"
    r"\{[a-z0-9_\-\s]*(option|title|problem|decision|author|date|status)[a-z0-9_\-\s]*\}|"
    r"\[[a-z0-9_\-\s]*(short title|title of|name of option|solved problem|YYYY-MM-DD|status)[a-z0-9_\-\s]*\])",
    re.IGNORECASE,
)

BOILERPLATE_FILE_PATTERNS = [
    "readme.md",
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
    "index.md",
    "examples.md",
    "tooling.md",
    "skill.md",
    "adr-guide.md",
]


def has_template_placeholders(text: str) -> bool:
    """Check if text contains unresolved template placeholders like ${...}, {{...}}, {title of option...}."""
    if not text:
        return False
    return bool(TEMPLATE_PLACEHOLDER_REGEX.search(text))


def is_boilerplate_filename(filename: str, rel_path: str = "") -> bool:
    """Check if a file or path matches boilerplate / template / changelog / release documentation."""
    fname_lower = filename.lower()
    path_lower = rel_path.lower() if rel_path else fname_lower

    for pattern in BOILERPLATE_FILE_PATTERNS:
        if pattern in fname_lower or pattern in path_lower:
            return True
    return False


def extract_markdown_section(text: str, header_pattern: str) -> str:
    """
    Extract full markdown section content up to the next same or higher level heading.
    Matches headings at start of line (e.g. \n# or \n##).
    """
    match = re.search(header_pattern, text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return ""
    start = match.end()
    remainder = text[start:]
    next_match = re.search(r"\n#{1,2}\s+", remainder)
    if next_match:
        section_text = remainder[: next_match.start()]
    else:
        section_text = remainder
    return section_text.strip()
