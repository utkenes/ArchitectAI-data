"""
Immutable Production Source Downloader and Raw Fixture Manager
"""

import subprocess
from pathlib import Path

from architectai_dataset_builder.sources.registry import SourceRegistry
from architectai_dataset_builder.utils.hashing import compute_sha256_file, compute_sha256_str


class ProductionSourceUnavailableError(Exception):
    """Raised when a required source cannot be fetched in production mode."""


class SourceDownloader:
    def __init__(self, data_dir: Path, registry: SourceRegistry) -> None:
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.registry = registry

    def fetch_source(self, source_id: str, mode: str = "production") -> Path:
        manifest = self.registry.get_manifest(source_id)
        if not manifest:
            raise ValueError(f"Unknown source_id: {source_id}")

        dest_dir = self.raw_dir / source_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        repo_url = manifest.origin.repository_url
        revision = manifest.version.revision or manifest.version.release_version or "main"
        manifest.version.requested_ref = revision

        is_git_repo = repo_url and repo_url.startswith("https://github.com/")

        if mode == "production" and is_git_repo and repo_url:
            success = self._fetch_git_repository(repo_url, revision, dest_dir)
            if not success:
                raise ProductionSourceUnavailableError(
                    f"Production fetch failed for source '{source_id}' from URL '{repo_url}' at revision '{revision}'."
                )

            # Resolve exact 40-character commit SHA
            resolved_sha = self._resolve_commit_sha(dest_dir)
            manifest.version.resolved_commit = resolved_sha or manifest.version.commit_sha
            manifest.version.commit_sha = manifest.version.resolved_commit
            manifest.notes = f"Mode: production | Ref: {revision} | Commit: {manifest.version.resolved_commit}"
        else:
            self._ensure_fixture_files(source_id, dest_dir)
            manifest.version.resolved_commit = manifest.version.commit_sha or "fixture_commit_00000000000000000000000000000000"
            manifest.notes = f"Mode: fixture | Local Directory: {dest_dir}"

        manifest.integrity["raw_sha256"] = self._compute_dir_hash(dest_dir)
        return dest_dir

    def _resolve_commit_sha(self, dest_dir: Path) -> str | None:
        try:
            res = subprocess.run(
                ["git", "-C", str(dest_dir), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except (subprocess.SubprocessError, OSError):
            pass
        return None

    def _fetch_git_repository(self, repo_url: str, revision: str, dest_dir: Path) -> bool:
        if any(dest_dir.iterdir()):
            return True

        try:
            res = subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", revision, repo_url, str(dest_dir)],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if res.returncode == 0:
                return True

            res_default = subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(dest_dir)],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            return res_default.returncode == 0
        except (subprocess.SubprocessError, OSError):
            return False

    def _compute_dir_hash(self, dir_path: Path) -> str:
        hashes = []
        for file_path in sorted(dir_path.rglob("*")):
            if file_path.is_file() and not file_path.name.startswith(".git"):
                hashes.append(f"{file_path.name}:{compute_sha256_file(file_path)}")
        return compute_sha256_str("\n".join(hashes))

    def _ensure_fixture_files(self, source_id: str, dest_dir: Path) -> None:
        if any(dest_dir.iterdir()):
            return

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

        elif source_id == "backstage_adrs":
            adr1 = dest_dir / "adr001-catalog-format.md"
            adr1.write_text(
                "# Default Catalog File Format\n\n"
                "## Context\n"
                "Backstage software catalog requires a standard serialization format for component entities.\n\n"
                "## Decision\n"
                "Adopt YAML schema with catalog-info.yaml as default descriptor file name.\n\n"
                "## Alternatives\n"
                "- JSON schema\n"
                "- TOML format\n\n"
                "## Consequences\n"
                "- Easy human readability\n"
                "- Native Kubernetes style CRD alignment\n",
                encoding="utf-8",
            )

        elif source_id == "k8s_keps":
            kep1 = dest_dir / "kep-1234-structured-logging.md"
            kep1.write_text(
                "# KEP-1234: Structured Logging System\n\n"
                "status: implementable\n\n"
                "## Summary\n"
                "Standardize JSON structured logging across all Kubernetes control plane components.\n\n"
                "## Motivation\n"
                "Text logs are hard to parse at scale for high volume telemetry systems.\n\n"
                "## Proposal\n"
                "Adopt klog structured logging API with contextual logging support.\n\n"
                "## Alternatives\n"
                "- Custom log parser sidecar\n"
                "- Zap logger directly\n\n"
                "## Risks and Mitigations\n"
                "- Log size increase: mitigate with log rate limiters.\n",
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
