"""
Architectural Relevance Classifier (Baseline Heuristic Filter)
"""

from typing import Dict, Any, List


class RelevanceFilter:
    def __init__(self, keywords: List[str], min_relevance_score: float = 0.60):
        self.keywords = [k.lower() for k in keywords]
        self.min_relevance_score = min_relevance_score

    def compute_score(self, text: str) -> float:
        if not text:
            return 0.0
        text_lower = text.lower()
        matched = sum(1 for kw in self.keywords if kw in text_lower)
        if matched == 0:
            return 0.0
        # Normalized score based on distinct keyword density
        score = min(1.0, matched / 5.0)
        return round(score, 2)

    def is_relevant(self, text: str) -> bool:
        return self.compute_score(text) >= self.min_relevance_score
