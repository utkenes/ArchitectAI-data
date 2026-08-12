"""
Evidence-Grounded Task Taxonomy Classifier

Rule: A task_type is assigned ONLY if the raw source evidence explicitly supports it.
If underlying evidence is missing or ambiguous, defaults safely to adr_reasoning or architecture_explanation.
"""

from typing import Dict, Any, List
from architectai_dataset_builder.models.canonical import TaskType


class TaskTaxonomyClassifier:
    def classify(self, parsed_record: Dict[str, Any]) -> TaskType:
        raw_text = parsed_record.get("raw_text", "").lower()
        options = parsed_record.get("options", [])
        alternatives = parsed_record.get("alternatives", [])
        positive = parsed_record.get("positive_consequences", [])
        negative = parsed_record.get("negative_consequences", [])
        plantuml = parsed_record.get("plantuml_text", "")

        # 1. Architecture Generation: explicit PlantUML or diagram code
        if plantuml or "@startuml" in raw_text:
            return TaskType.ARCHITECTURE_GENERATION

        # 2. Tradeoff Analysis: explicit positive vs negative consequences or options comparison
        if (positive and negative) or (len(options) > 1 and "tradeoff" in raw_text):
            return TaskType.TRADEOFF_ANALYSIS

        # 3. Quality Attribute Reasoning: explicit mention of quality attributes
        if any(kw in raw_text for kw in ["availability", "latency", "throughput", "reliability", "scalability"]):
            if "quality attribute" in raw_text or "nfr" in raw_text:
                return TaskType.QUALITY_ATTRIBUTE_REASONING

        # 4. ADR Reasoning: explicit decision outcome / decision section present
        if parsed_record.get("decision_outcome") or parsed_record.get("decision"):
            return TaskType.ADR_REASONING

        # 5. Default safe fallback without force-fitting
        return TaskType.ARCHITECTURE_EXPLANATION
