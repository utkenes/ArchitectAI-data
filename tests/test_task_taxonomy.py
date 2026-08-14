from architectai_dataset_builder.models.canonical import TaskType
from architectai_dataset_builder.normalizers.task_taxonomy import TaskTaxonomyClassifier


def test_taxonomy_classification_with_evidence():
    classifier = TaskTaxonomyClassifier()

    # 1. Tradeoff Analysis with explicit comparison evidence
    rec_tradeoff = {
        "context": "We need to choose a database model.",
        "options": ["Relational DB", "NoSQL Document Store"],
        "positive_consequences": ["ACID compliance"],
        "negative_consequences": ["Scalability limits"],
        "raw_text": "Evaluating the trade-off between Relational DB vs NoSQL Document Store.",
    }
    assert classifier.classify(rec_tradeoff) == TaskType.TRADEOFF_ANALYSIS

    # 2. Architecture Generation
    rec_arch_gen = {
        "context": "System requirements",
        "plantuml_text": "@startuml\n[Component]\n@enduml",
        "raw_text": "@startuml",
    }
    assert classifier.classify(rec_arch_gen) == TaskType.ARCHITECTURE_GENERATION

    # 3. Safe fallback without decision evidence
    rec_generic = {"context": "Generic overview text without decision outcome", "raw_text": "plain explanation of architecture"}
    assert classifier.classify(rec_generic) == TaskType.ARCHITECTURE_EXPLANATION


def test_false_scaling_keyword_rejected():
    classifier = TaskTaxonomyClassifier()
    # Sentence containing 'scale' without capacity driver or architectural response
    rec = {
        "context": "APIs allow ecosystems to scale quickly across partners.",
        "decision": "Use standard OpenAPI specifications.",
        "raw_text": "APIs allow ecosystems to scale quickly across partners. Decision: Use standard OpenAPI specifications.",
    }
    task = classifier.classify(rec)
    assert task != TaskType.SCALING_REASONING
    assert task == TaskType.ADR_REASONING


def test_real_scaling_example_accepted():
    classifier = TaskTaxonomyClassifier()
    rec = {
        "context": "High workload growth caused severe partition pressure on primary database nodes.",
        "decision": "Implement database sharding and horizontal pod autoscaler to handle high throughput.",
        "raw_text": "High workload growth caused severe partition pressure on primary database nodes. Decision: Implement database sharding and horizontal pod autoscaler to handle high throughput.",
    }
    assert classifier.classify(rec) == TaskType.SCALING_REASONING


def test_false_technology_selection_rejected():
    classifier = TaskTaxonomyClassifier()
    rec = {
        "context": "This project utilizes an internal framework.",
        "decision": "Follow standard coding standards.",
        "raw_text": "This project utilizes an internal framework. Decision: Follow standard coding standards.",
    }
    task = classifier.classify(rec)
    assert task != TaskType.TECHNOLOGY_SELECTION


def test_real_technology_selection_accepted():
    classifier = TaskTaxonomyClassifier()
    rec = {
        "context": "We need an inter-service communication protocol.",
        "options": ["gRPC", "REST"],
        "decision": "Selected gRPC over REST after benchmark evaluation.",
        "raw_text": "We evaluated gRPC versus REST for inter-service communication and selected gRPC over REST.",
    }
    assert classifier.classify(rec) == TaskType.TECHNOLOGY_SELECTION
