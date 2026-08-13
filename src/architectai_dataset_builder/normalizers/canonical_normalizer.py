"""
Canonical Sample Normalizer with Evidence Grounding and Composite Group IDs
"""

from datetime import UTC, datetime
from typing import Any

from architectai_dataset_builder.models.canonical import (
    Alternative,
    ArchitectAISample,
    RecommendedArchitecture,
    ReviewInfo,
    ReviewStatus,
    SourceMetadata,
)
from architectai_dataset_builder.models.evidence import EvidenceItem, EvidenceType
from architectai_dataset_builder.models.manifest import SourceManifest
from architectai_dataset_builder.normalizers.task_taxonomy import TaskTaxonomyClassifier
from architectai_dataset_builder.utils.hashing import compute_sha256_str


class CanonicalNormalizer:
    def __init__(self) -> None:
        self.taxonomy_classifier = TaskTaxonomyClassifier()

    def normalize(
        self,
        parsed_record: dict[str, Any],
        manifest: SourceManifest,
        split: str | None = None,
    ) -> ArchitectAISample:
        sample_id = parsed_record["sample_id"]
        raw_hash = parsed_record["raw_sha256"]
        raw_text = parsed_record.get("raw_text", "")
        norm_hash = compute_sha256_str(raw_text)

        # 1. Grounded Task Taxonomy
        task_type = self.taxonomy_classifier.classify(parsed_record)

        # 2. Composite Group ID (Prevents ID collisions across repos)
        project_id = parsed_record.get("project_id") or "default"
        record_id = parsed_record.get("record_id", sample_id)
        group_id = f"group_{manifest.source_id}_{project_id}_{record_id}"

        # 3. Source Provenance Metadata
        kep_status = parsed_record.get("kep_status")
        source_meta = SourceMetadata(
            source_id=manifest.source_id,
            source_name=manifest.name,
            source_url=manifest.origin.repository_url,
            source_version=manifest.version.revision or manifest.version.release_version,
            source_commit_sha=manifest.version.commit_sha or manifest.version.resolved_commit,
            requested_ref=manifest.version.requested_ref or manifest.version.revision or "main",
            resolved_commit=manifest.version.resolved_commit,
            source_file_path=parsed_record.get("file_name", "unknown"),
            source_record_id=record_id,
            project_id=project_id,
            group_id=group_id,
            provenance_type="real_world",
            license_id=manifest.license.spdx_id,
            license_verified=manifest.license.verified,
            raw_sha256=raw_hash,
            normalized_sha256=norm_hash,
            split=split,
            kep_status=kep_status,
            created_at=datetime.now(UTC).isoformat(),
        )

        # 4. Grounded Context & Facts
        context = parsed_record.get("context") or parsed_record.get("summary") or parsed_record.get("requirements_text") or parsed_record.get("title") or "Architectural Scenario"
        scenario = context.strip()
        if kep_status:
            scenario = f"[KEP Status: {kep_status.upper()}] {scenario}"

        facts = []
        if parsed_record.get("title"):
            facts.append(
                EvidenceItem(
                    value=f"Title: {parsed_record['title']}",
                    evidence_type=EvidenceType.EXPLICIT,
                )
            )

        # 5. Drivers & Constraints
        drivers = []
        for d in parsed_record.get("drivers", []):
            drivers.append(EvidenceItem(value=d, evidence_type=EvidenceType.EXPLICIT))

        # 6. Decisions & Consequences (Exclude "Not explicitly stated")
        decisions = []
        dec_val = parsed_record.get("decision_outcome") or parsed_record.get("decision") or parsed_record.get("proposal")
        if dec_val and dec_val.strip().lower() != "not explicitly stated":
            decisions.append(EvidenceItem(value=dec_val.strip(), evidence_type=EvidenceType.EXPLICIT))

        tradeoffs = []
        for pos in parsed_record.get("positive_consequences", []):
            tradeoffs.append(
                EvidenceItem(value=f"Advantage: {pos}", evidence_type=EvidenceType.EXPLICIT)
            )
        for neg in parsed_record.get("negative_consequences", []):
            tradeoffs.append(
                EvidenceItem(value=f"Disadvantage: {neg}", evidence_type=EvidenceType.EXPLICIT)
            )
        for t in parsed_record.get("tradeoffs", []):
            tradeoffs.append(
                EvidenceItem(value=f"Risk/Trade-off: {t}", evidence_type=EvidenceType.EXPLICIT)
            )

        # 7. Alternatives
        alternatives = []
        for opt in parsed_record.get("options", []) or parsed_record.get("alternatives", []):
            alternatives.append(Alternative(option=opt))

        # 8. Recommended Architecture
        rec_arch = None
        if dec_val and dec_val.strip().lower() != "not explicitly stated":
            rec_arch = RecommendedArchitecture(
                summary=dec_val.strip(),
                components=[c for c in parsed_record.get("options", [])],
            )
        elif parsed_record.get("plantuml_text"):
            rec_arch = RecommendedArchitecture(
                summary="PlantUML Architecture Diagram Specification",
                diagram_plantuml=parsed_record["plantuml_text"],
            )

        # 9. Final Answer (Strictly exclude "Not explicitly stated")
        final_answer = None
        if dec_val and dec_val.strip().lower() != "not explicitly stated":
            final_answer = f"Decision / Proposal:\n{dec_val.strip()}"
            if parsed_record.get("rationale") and parsed_record["rationale"].strip().lower() != "not explicitly stated":
                final_answer += f"\n\nRationale:\n{parsed_record['rationale'].strip()}"
        elif parsed_record.get("plantuml_text"):
            final_answer = f"Architecture Design (PlantUML):\n```plantuml\n{parsed_record['plantuml_text'].strip()}\n```"

        return ArchitectAISample(
            id=sample_id,
            source=source_meta,
            scenario=scenario,
            task_type=task_type,
            facts=facts,
            architecture_drivers=drivers,
            recommended_architecture=rec_arch,
            alternatives=alternatives,
            decisions=decisions,
            tradeoffs=tradeoffs,
            final_answer=final_answer,
            review=ReviewInfo(status=ReviewStatus.UNREVIEWED),
        )
