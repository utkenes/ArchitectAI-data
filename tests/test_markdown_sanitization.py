from architectai_dataset_builder.utils.markdown import (
    extract_markdown_section,
    extract_structured_items,
    sanitize_markdown,
)


def test_sanitize_markdown_html_comments():
    raw = """
# Decision Title

<!--
What other approaches did you consider?
Describe why they were rejected.
-->

The decision is to use PostgreSQL.
"""
    clean = sanitize_markdown(raw)
    assert "<!--" not in clean
    assert "-->" not in clean
    assert "What other approaches" not in clean
    assert "The decision is to use PostgreSQL." in clean


def test_heading_level_aware_extraction():
    md = """
## Context
This is context text.

### Alternatives
Option A is primary.
Option B is secondary.

### Risks
Risk 1: High latency.
"""
    alt_section = extract_markdown_section(md, ["alternatives"])
    assert "Option A" in alt_section
    assert "Option B" in alt_section
    assert "Risks" not in alt_section
    assert "High latency" not in alt_section


def test_structured_list_extraction_bullets():
    text = """
- Use Redis for caching
- Use PostgreSQL for persistence
* Optional: Memory Cache
"""
    items = extract_structured_items(text)
    assert len(items) == 3
    assert items[0] == "Use Redis for caching"
    assert items[1] == "Use PostgreSQL for persistence"
    assert items[2] == "Optional: Memory Cache"


def test_structured_list_extraction_prose():
    text = "We evaluated several approaches and selected PostgreSQL due to strong transactional guarantees."
    items = extract_structured_items(text)
    assert len(items) == 1
    assert items[0] == text
