"""
Immutable Source Downloader and Raw Fixture Manager
"""

from pathlib import Path
from typing import Dict, Any
from architectai_dataset_builder.sources.registry import SourceRegistry
from architectai_dataset_builder.utils.hashing import compute_sha256_file, compute_sha256_str


class SourceDownloader:
    def __init__(self, data_dir: Path, registry: SourceRegistry):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.registry = registry

    def fetch_source(self, source_id: str, create_fixtures_if_missing: bool = True) -> Path:
        manifest = self.registry.get_manifest(source_id)
        if not manifest:
            raise ValueError(f"Unknown source_id: {source_id}")

        dest_dir = self.raw_dir / source_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        # In production pipeline, if git repo URL is present, this would git clone / fetch pinned commit SHA.
        # For offline reproducible V1 bootstrap/fixtures, we ensure raw files exist in raw_dir and compute raw_sha256.
        if create_fixtures_if_missing:
            self._ensure_fixture_files(source_id, dest_dir)

        # Compute SHA-256 for all raw files in dest_dir
        manifest.integrity["raw_sha256"] = self._compute_dir_hash(dest_dir)
        return dest_dir

    def _compute_dir_hash(self, dir_path: Path) -> str:
        hashes = []
        for file_path in sorted(dir_path.rglob("*")):
            if file_path.is_file():
                hashes.append(f"{file_path.name}:{compute_sha256_file(file_path)}")
        return compute_sha256_str("\n".join(hashes))

    def _ensure_fixture_files(self, source_id: str, dest_dir: Path) -> None:
        if any(dest_dir.iterdir()):
            return  # Already populated

        if source_id == "madr":
            adr1 = dest_dir / "0001-use-madr.md"
            adr1.write_text(
                "# Use Markdown Architectural Decision Records (MADR)\n\n"
                "## Context and Problem Statement\n"
                "We need a lightweight, standard format for storing architectural decisions in git.\n\n"
                "## Decision Drivers\n"
                "* Text-based format versionable in Git\n"
                "* Human-readable and markdown compliant\n\n"
                "## Considered Options\n"
                "* MADR format\n"
                "* Plain text files\n"
                "* Wiki pages\n\n"
                "## Decision Outcome\n"
                "Chosen option: MADR format, because it structures decisions cleanly with context, options, and consequences.\n\n"
                "## Positive Consequences\n"
                "* Decisions are stored alongside code\n"
                "* Clear template structure\n\n"
                "## Negative Consequences\n"
                "* Maintenance overhead for team\n",
                encoding="utf-8",
            )
            adr2 = dest_dir / "0002-use-event-driven-architecture.md"
            adr2.write_text(
                "# Use Event-Driven Architecture for Order Processing\n\n"
                "## Context and Problem Statement\n"
                "Order fulfillment service needs to decoupled from payment and inventory updates under high throughput.\n\n"
                "## Decision Drivers\n"
                "* High availability\n"
                "* Decoupled domain services\n\n"
                "## Considered Options\n"
                "* Event-driven architecture with RabbitMQ\n"
                "* Direct REST API calls\n\n"
                "## Decision Outcome\n"
                "Chosen option: Event-driven architecture with RabbitMQ, because it provides asynchronous message queuing and failure recovery.\n\n"
                "## Positive Consequences\n"
                "* System components fail independently\n"
                "* Improved scaling under spikes\n\n"
                "## Negative Consequences\n"
                "* Eventual consistency complexity\n",
                encoding="utf-8",
            )

        elif source_id == "opendatahub_adr":
            adr1 = dest_dir / "0001-pipeline-orchestration.md"
            adr1.write_text(
                "# Pipeline Orchestration Engine\n\n"
                "## Status\n"
                "Accepted\n\n"
                "## Context\n"
                "Open Data Hub requires a reliable workflow orchestrator for machine learning pipelines.\n\n"
                "## Alternatives Considered\n"
                "- Kubeflow Pipelines\n"
                "- Apache Airflow\n"
                "- Argo Workflows\n\n"
                "## Decision\n"
                "We decide to adopt Tekton and Kubeflow Pipelines for Kubernetes native workflow execution.\n\n"
                "## Rationale\n"
                "Kubeflow offers native UI integration while Tekton ensures cloud-native CRD based execution.\n",
                encoding="utf-8",
            )

        elif source_id == "r2abench":
            prd1 = dest_dir / "proj_001_ecommerce_req.txt"
            prd1.write_text(
                "Project: proj_001_ecommerce\n"
                "System Requirements:\n"
                "1. The system must support high availability shopping cart service.\n"
                "2. System must process payment processing asynchronously via event queues.\n"
                "3. Database consistency must enforce relational ACID for payment audit logs.\n",
                encoding="utf-8",
            )
            arch1 = dest_dir / "proj_001_ecommerce_arch.puml"
            arch1.write_text(
                "@startuml\n"
                "package ShoppingCart {\n"
                "  [Cart API] --> [Redis Cache]\n"
                "  [Cart API] --> [PostgreSQL DB]\n"
                "}\n"
                "@enduml\n",
                encoding="utf-8",
            )

        elif source_id == "sake":
            sake_fixture = dest_dir / "sake_questions.json"
            sake_fixture.write_text(
                '[\n'
                '  {\n'
                '    "id": "sake_q001",\n'
                '    "question": "Which architectural pattern decouples producers and consumers asynchronously?",\n'
                '    "options": {"A": "Monolith", "B": "Publish-Subscribe", "C": "Shared Database", "D": "Direct RPC"},\n'
                '    "correct_answer": "B",\n'
                '    "explanation": "Publish-Subscribe decouples message senders from receivers using event brokers."\n'
                '  }\n'
                ']\n',
                encoding="utf-8",
            )

        elif source_id == "cake":
            cake_fixture = dest_dir / "cake_eval.json"
            cake_fixture.write_text(
                '[\n'
                '  {\n'
                '    "id": "cake_q001",\n'
                '    "prompt": "Explain trade-offs between monolithic database and database-per-service pattern.",\n'
                '    "reference_answer": "Monolithic DB provides easy ACID transactions; DB-per-service improves service isolation but introduces distributed consistency challenges.",\n'
                '    "grading_rubric": ["Isolation", "ACID vs Eventual Consistency", "Operational Complexity"]\n'
                '  }\n'
                ']\n',
                encoding="utf-8",
            )

        elif source_id == "archbench":
            arch_fixture = dest_dir / "archbench_tasks.json"
            arch_fixture.write_text(
                '[\n'
                '  {\n'
                '    "id": "arch_t001",\n'
                '    "requirements": "Design a resilient telemetry pipeline supporting 100k events/sec.",\n'
                '    "constraints": ["Sub-second latency", "Zero event loss"],\n'
                '    "expected_components": ["Kafka", "Flink", "ClickHouse"],\n'
                '    "reference_solution": "Use Kafka for ingress queuing, Flink for real-time aggregation, and ClickHouse for analytical queries."\n'
                '  }\n'
                ']\n',
                encoding="utf-8",
            )
