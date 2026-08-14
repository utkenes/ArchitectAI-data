"""
Evidence-Grounded Task Taxonomy Classifier V2.1

Rule: A task_type is assigned ONLY if the raw source evidence explicitly satisfies
the evidence contract for that task. If evidence is missing or ambiguous,
the classifier falls back conservatively to adr_reasoning or architecture_explanation.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from architectai_dataset_builder.models.canonical import TaskType


@dataclass
class ClassificationResult:
    task_type: TaskType
    confidence: float
    evidence: list[str] = field(default_factory=list)


SCALING_DRIVERS = [
    r"\bthroughput\b",
    r"\bconcurrent users\b",
    r"\bdata volume\b",
    r"\bqueue depth\b",
    r"\bworkload growth\b",
    r"\bhorizontal capacity\b",
    r"\bpartition pressure\b",
    r"\bresource saturation\b",
    r"\brequest volume\b",
    r"\bpeak load\b",
    r"\bscale out\b",
    r"\bhigh load\b",
]

SCALING_RESPONSES = [
    r"\bpartitioning\b",
    r"\bsharding\b",
    r"\bcaching\b",
    r"\bqueueing\b",
    r"\bload balancing\b",
    r"\breplication\b",
    r"\bautoscaling\b",
    r"\bworkload isolation\b",
    r"\bhorizontal pod autoscaler\b",
    r"\bhpa\b",
]

CONCRETE_TECH_TERMS = [
    r"\bpostgresql\b",
    r"\bmysql\b",
    r"\bmongodb\b",
    r"\bredis\b",
    r"\bkafka\b",
    r"\brabbitmq\b",
    r"\bgrpc\b",
    r"\brest\b",
    r"\bgraphql\b",
    r"\bdocker\b",
    r"\bkubernetes\b",
    r"\benvoy\b",
    r"\bnginx\b",
    r"\bprometheus\b",
    r"\belasticsearch\b",
    r"\bistio\b",
    r"\baws\b",
    r"\bazure\b",
    r"\bgcp\b",
    r"\bspanner\b",
    r"\bcockroachdb\b",
]

GENERIC_TECH_CATEGORIES = [
    r"\bvendor\b",
    r"\bdatabase\b",
    r"\bframework\b",
    r"\btechnology\b",
    r"\bplatform\b",
    r"\blibrary\b",
    r"\btool\b",
    r"\bdriver\b",
]

TECH_COMPARISONS = [
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bcompared to\b",
    r"\bcompared with\b",
    r"\bevaluated\b",
    r"\bevaluating\b",
    r"\bbenchmark\b",
    r"\bselected over\b",
    r"\brejected in favor of\b",
    r"\balternative to\b",
    r"\btradeoff between\b",
]

QUALITY_ATTRIBUTES = [
    r"\bavailability\b",
    r"\blatency\b",
    r"\breliability\b",
    r"\bsecurity\b",
    r"\bresilience\b",
    r"\bconsistency\b",
    r"\bperformance\b",
]

QUALITY_REASONING_INDICATORS = [
    r"\bnfr\b",
    r"\bnon-functional requirement\b",
    r"\bsla\b",
    r"\bslo\b",
    r"\bquality attribute\b",
    r"\btrade-off between\b",
    r"\bguarantee\b",
    r"\bavailability target\b",
    r"\blatency budget\b",
]

TRADEOFF_INDICATORS = [
    r"\btrade-off\b",
    r"\btradeoff\b",
    r"\bpros and cons\b",
    r"\badvantages and disadvantages\b",
    r"\bcost-benefit\b",
    r"\bmitigation\b",
]


class TaskTaxonomyClassifier:
    def classify(self, parsed_record: dict[str, Any]) -> TaskType:
        return self.classify_with_evidence(parsed_record).task_type

    def classify_with_evidence(self, parsed_record: dict[str, Any]) -> ClassificationResult:
        raw_text = parsed_record.get("raw_text", "").lower()
        options = parsed_record.get("options", []) or parsed_record.get("alternatives", [])
        positive = parsed_record.get("positive_consequences", [])
        negative = parsed_record.get("negative_consequences", [])
        plantuml = parsed_record.get("plantuml_text", "")
        summary = parsed_record.get("summary", "").lower()
        motivation = parsed_record.get("motivation", "").lower()
        context = parsed_record.get("context", "").lower()
        decision = (
            parsed_record.get("decision_outcome")
            or parsed_record.get("decision")
            or parsed_record.get("proposal")
            or ""
        ).lower()

        # 1. Architecture Generation: explicit PlantUML or diagram code
        if plantuml or "@startuml" in raw_text:
            return ClassificationResult(
                task_type=TaskType.ARCHITECTURE_GENERATION,
                confidence=1.0,
                evidence=["Explicit PlantUML architecture specification"],
            )

        # 2. Scaling Reasoning: Real capacity/load driver AND architectural response
        driver_matches = [p for p in SCALING_DRIVERS if re.search(p, raw_text)]
        response_matches = [p for p in SCALING_RESPONSES if re.search(p, raw_text)]
        if driver_matches and response_matches:
            return ClassificationResult(
                task_type=TaskType.SCALING_REASONING,
                confidence=0.91,
                evidence=[
                    f"Scaling driver matched: {', '.join(driver_matches[:2])}",
                    f"Architectural response matched: {', '.join(response_matches[:2])}",
                ],
            )

        # 3. Technology Selection: Concrete tech terms + (comparison or 2+ options), OR Generic category + comparison
        concrete_matches = [p for p in CONCRETE_TECH_TERMS if re.search(p, raw_text)]
        generic_matches = [p for p in GENERIC_TECH_CATEGORIES if re.search(p, raw_text)]
        comp_matches = [p for p in TECH_COMPARISONS if re.search(p, raw_text)]

        is_concrete_tech_selection = bool(concrete_matches) and bool(comp_matches or len(options) >= 2)
        is_generic_tech_selection = bool(generic_matches) and bool(comp_matches)

        if is_concrete_tech_selection or is_generic_tech_selection:
            matched_terms = concrete_matches or generic_matches
            return ClassificationResult(
                task_type=TaskType.TECHNOLOGY_SELECTION,
                confidence=0.88,
                evidence=[
                    f"Technology terms: {', '.join(matched_terms[:2])}",
                    f"Comparison evidence: {comp_matches[0] if comp_matches else 'multiple options'}",
                ],
            )

        # 4. Quality Attribute Reasoning: Explicit NFR/quality attribute AND architectural reasoning
        qa_matches = [p for p in QUALITY_ATTRIBUTES if re.search(p, raw_text)]
        ind_matches = [p for p in QUALITY_REASONING_INDICATORS if re.search(p, raw_text)]
        if qa_matches and ind_matches:
            return ClassificationResult(
                task_type=TaskType.QUALITY_ATTRIBUTE_REASONING,
                confidence=0.86,
                evidence=[
                    f"Quality attribute: {', '.join(qa_matches[:2])}",
                    f"Reasoning indicator: {ind_matches[0]}",
                ],
            )

        # 5. Tradeoff Analysis: Explicit tradeoffs/consequences evidence or tradeoff comparison indicators
        tradeoffs = parsed_record.get("tradeoffs", []) or parsed_record.get("consequences", [])
        tradeoff_matches = [p for p in TRADEOFF_INDICATORS if re.search(p, raw_text)]

        has_explicit_tradeoff_section = len(tradeoffs) > 0 or (len(positive) > 0 and len(negative) > 0)
        has_tradeoff_indicator = bool(tradeoff_matches) and (len(options) >= 1 or len(tradeoffs) >= 1)

        if has_explicit_tradeoff_section or has_tradeoff_indicator:
            return ClassificationResult(
                task_type=TaskType.TRADEOFF_ANALYSIS,
                confidence=0.85,
                evidence=[
                    f"Tradeoff/consequence items count: {len(tradeoffs)}, pos={len(positive)}, neg={len(negative)}",
                    f"Tradeoff indicator: {tradeoff_matches[0] if tradeoff_matches else 'explicit section'}",
                ],
            )

        # 6. ADR Reasoning: Architectural context + explicit decision outcome
        is_decision_valid = bool(decision) and decision != "not explicitly stated" and len(decision.strip()) >= 15
        is_context_valid = bool(context or summary or motivation) and len((context or summary or motivation).strip()) >= 20
        if is_decision_valid and is_context_valid:
            return ClassificationResult(
                task_type=TaskType.ADR_REASONING,
                confidence=0.80,
                evidence=["Explicit architectural context and decision outcome present"],
            )

        # 7. Architecture Explanation: Overview without explicit decision outcome
        return ClassificationResult(
            task_type=TaskType.ARCHITECTURE_EXPLANATION,
            confidence=0.70,
            evidence=["Architectural context present without explicit decision evidence"],
        )
