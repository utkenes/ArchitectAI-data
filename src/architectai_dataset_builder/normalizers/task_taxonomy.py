"""
Evidence-Grounded Task Taxonomy Classifier

Rule: A task_type is assigned ONLY if the raw source evidence explicitly supports it.
If underlying evidence is missing or ambiguous, defaults safely to adr_reasoning or architecture_explanation.
"""

from typing import Any

from architectai_dataset_builder.models.canonical import TaskType


class TaskTaxonomyClassifier:
    def classify(self, parsed_record: dict[str, Any]) -> TaskType:
        raw_text = parsed_record.get("raw_text", "").lower()
        options = parsed_record.get("options", []) or parsed_record.get("alternatives", [])
        positive = parsed_record.get("positive_consequences", [])
        negative = parsed_record.get("negative_consequences", [])
        plantuml = parsed_record.get("plantuml_text", "")
        summary = parsed_record.get("summary", "").lower()
        motivation = parsed_record.get("motivation", "").lower()

        # 1. Architecture Generation: explicit PlantUML or diagram code
        if plantuml or "@startuml" in raw_text:
            return TaskType.ARCHITECTURE_GENERATION

        # 2. Technology Selection: explicit tech stack evaluation or tool comparison
        if ("technology" in raw_text or "framework" in raw_text or "vendor" in raw_text) and (
            len(options) > 1 or "select" in raw_text or "evaluate" in raw_text
        ):
            return TaskType.TECHNOLOGY_SELECTION

        # 3. Scaling Reasoning: explicit throughput, scaling, volume, or load drivers
        if any(kw in raw_text for kw in ["scaling", "scale", "high throughput", "load balancing", "horizontal pod autoscaler", "sharding"]):
            return TaskType.SCALING_REASONING

        # 4. Quality Attribute Reasoning: explicit mention of quality attributes or NFRs
        if any(kw in raw_text for kw in ["availability", "latency", "throughput", "reliability", "scalability", "resilience"]) and (
            "quality attribute" in raw_text or "nfr" in raw_text or "non-functional requirement" in raw_text or "sla" in raw_text
        ):
            return TaskType.QUALITY_ATTRIBUTE_REASONING

        # 5. Tradeoff Analysis: explicit positive vs negative consequences or options comparison
        if (positive and negative) or (len(options) > 1 and ("tradeoff" in raw_text or "trade-off" in raw_text or "risk" in raw_text)):
            return TaskType.TRADEOFF_ANALYSIS

        # 6. Architecture Explanation: summary or motivation overview without explicit decision outcome
        if (summary or motivation) and not parsed_record.get("decision_outcome") and not parsed_record.get("decision"):
            return TaskType.ARCHITECTURE_EXPLANATION

        # 7. ADR Reasoning: explicit decision outcome / decision section present
        if parsed_record.get("decision_outcome") or parsed_record.get("decision") or parsed_record.get("proposal"):
            return TaskType.ADR_REASONING

        # 8. Default safe fallback without force-fitting
        return TaskType.ARCHITECTURE_EXPLANATION
