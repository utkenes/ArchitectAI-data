"""
Corpus V1 Freeze Manifest & SHA-256 Artifact Fingerprint Exporter
"""

from pathlib import Path
from typing import Dict, Any
from architectai_dataset_builder.utils.hashing import compute_sha256_file
from architectai_dataset_builder.utils.io import write_jsonl


class CorpusManifestExporter:
    def __init__(self, export_dir: Path) -> None:
        self.export_dir = Path(export_dir)

    def export_corpus_manifest(
        self,
        build_id: str,
        sources_summary: Dict[str, Dict[str, Any]],
        sample_counts: Dict[str, int],
        output_file: Path,
    ) -> Dict[str, Any]:
        # Non-recursive SHA-256 artifact fingerprints registry
        artifact_files = [
            "train_sft.jsonl",
            "validation_sft.jsonl",
            "silver.jsonl",
            "gold.jsonl",
            "eval_manifest.json",
            "quality_review_samples.jsonl",
            "gold_review_candidates.jsonl",
        ]

        artifact_hashes: Dict[str, str] = {}
        for fname in artifact_files:
            fpath = self.export_dir / fname
            if fpath.exists():
                artifact_hashes[fname] = f"sha256:{compute_sha256_file(fpath)}"

        # Include evaluation benchmark exports under eval/
        eval_dir = self.export_dir / "eval"
        if eval_dir.exists():
            for ef in sorted(eval_dir.glob("*.jsonl")):
                rel_name = f"eval/{ef.name}"
                artifact_hashes[rel_name] = f"sha256:{compute_sha256_file(ef)}"

        manifest_content: Dict[str, Any] = {
            "corpus_name": "ArchitectAI Training Corpus V1",
            "corpus_version": "1.0.0",
            "build_id": build_id,
            "quality_gate_status": "READY_WITH_WARNINGS",
            "source_revisions": sources_summary,
            "sample_counts": sample_counts,
            "artifact_hashes": artifact_hashes,
        }

        write_jsonl([manifest_content], output_file)
        return manifest_content
