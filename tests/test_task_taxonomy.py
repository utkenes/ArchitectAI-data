import pytest
from architectai_dataset_builder.normalizers.task_taxonomy import TaskTaxonomyClassifier
from architectai_dataset_builder.models.canonical import TaskType


def test_taxonomy_classification_with_evidence():
    classifier = TaskTaxonomyClassifier()

    # 1. Tradeoff Analysis
    rec_tradeoff = {
        "context": "Context text",
        "options": ["Opt 1", "Opt 2"],
        "positive_consequences": ["Advantage 1"],
        "negative_consequences": ["Disadvantage 1"],
        "raw_text": "tradeoff analysis",
    }
    assert classifier.classify(rec_tradeoff) == TaskType.TRADEOFF_ANALYSIS

    # 2. Architecture Generation
    rec_arch_gen = {
        "context": "System requirements",
        "plantuml_text": "@startuml\n[Component]\n@enduml",
        "raw_text": "@startuml",
    }
    assert classifier.classify(rec_arch_gen) == TaskType.ARCHITECTURE_GENERATION

    # 3. Safe fallback without evidence
    rec_generic = {"context": "Generic text without specific headers", "raw_text": "plain explanation"}
    assert classifier.classify(rec_generic) == TaskType.ARCHITECTURE_EXPLANATION
