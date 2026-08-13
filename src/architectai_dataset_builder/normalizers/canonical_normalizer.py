"""
Canonical Sample Normalizer with Evidence Grounding and Composite Group IDs
"""

from datetime import datetime, timezone
from typing import Any, Optional
from architectai_dataset_builder.models.canonical import (
    ArchitectAISample,
    SourceMetadata,
    Alternative,
    RecommendedArchitecture,
    ReviewInfo,
    ReviewStatus,
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
        split: Optional[str] = None,
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
            source_commit_sha=manifest.version.commit_sha,
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
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # 4. Grounded Context & Facts
        scenario = parsed_record.get("context") or parsed_record.get("summary") or parsed_record.get("title") or "Architectural Scenario"
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

        # 6. Decisions & Consequences
        decisions = []
        dec_val = parsed_record.get("decision_outcome") or parsed_record.get("decision") or parsed_record.get("proposal")
        if dec_val:
            decisions.append(EvidenceItem(value=dec_val, evidence_type=EvidenceType.EXPLICIT))

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
        if dec_val or parsed_record.get("plantuml_text"):
            rec_arch = RecommendedArchitecture(
                summary=dec_val or "Architecture Design",
                components=[c for c in parsed_record.get("options", [])],
            )

        # 9. Final Answer
        final_answer = None
        if dec_val:
            final_answer = f"Decision / Proposal: {dec_val}"
            if parsed_record.get("rationale"):
                final_answer += f"\n\nRationale: {parsed_record['rationale']}"

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
