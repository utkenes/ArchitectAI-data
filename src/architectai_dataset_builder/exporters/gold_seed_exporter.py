"""
Gold Seed Candidate Exporter V2 with Stratified Candidate Selection & Full SFT Response Payload
"""

from collections import Counter
from pathlib import Path
from typing import Any

from architectai_dataset_builder.exporters.sft_formatter import SFTFormatter
from architectai_dataset_builder.models.canonical import ArchitectAISample
from architectai_dataset_builder.utils.hashing import hash_string_to_int
from architectai_dataset_builder.utils.io import write_jsonl


class GoldSeedExporter:
    def __init__(self, export_dir: Path) -> None:
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.sft_formatter = SFTFormatter()

    def export_review_candidates(
        self,
        samples: list[ArchitectAISample],
        target_candidate_count: int = 50,
        max_per_source_ratio: float = 0.50,
    ) -> Path:
        """
        Deterministically selects a stratified pool of candidates for manual review.
        Includes full SFT assistant response payload and complete evidence dictionary.
        """
        # Group by (task_type, source_id)
        strata: dict[tuple[str, str], list[ArchitectAISample]] = {}
        for sample in samples:
            key = (sample.task_type.value, sample.source.source_id)
            if key not in strata:
                strata[key] = []
            strata[key].append(sample)

        # Sort each stratum by deterministic score (evidence richness + scenario length + hash stability)
        for key, s_list in strata.items():
            s_list.sort(
                key=lambda s: (
                    -(
                        len(s.decisions) * 3
                        + len(s.tradeoffs) * 2
                        + len(s.alternatives) * 2
                        + len(s.facts)
                    ),
                    -len(s.scenario),
                    hash_string_to_int(s.id),
                )
            )

        selected_samples: list[ArchitectAISample] = []
        seen_group_ids: set[str] = set()
        source_counts: Counter[str] = Counter()

        max_from_single_source = max(1, int(target_candidate_count * max_per_source_ratio))

        # Round-robin selection across strata to ensure even task x source representation
        strata_keys = sorted(strata.keys())
        pass_idx = 0

        while len(selected_samples) < target_candidate_count and pass_idx < 100:
            added_in_pass = False
            for key in strata_keys:
                if len(selected_samples) >= target_candidate_count:
                    break
                s_list = strata[key]
                for sample in list(s_list):
                    gid = sample.source.group_id or sample.id
                    sid = sample.source.source_id

                    if gid in seen_group_ids:
                        continue
                    if source_counts[sid] >= max_from_single_source and len(strata_keys) > 1:
                        continue

                    # Check that SFT response is valid and non-empty
                    sft_resp = self.sft_formatter.compose_grounded_response(sample)
                    if not sft_resp or "not explicitly stated" in sft_resp.lower():
                        continue

                    selected_samples.append(sample)
                    seen_group_ids.add(gid)
                    source_counts[sid] += 1
                    s_list.remove(sample)
                    added_in_pass = True
                    break
            if not added_in_pass:
                # If constraints blocked further additions, relax source cap
                max_from_single_source += 5
            pass_idx += 1

        candidates: list[dict[str, Any]] = []
        for s in selected_samples:
            sft_response = self.sft_formatter.compose_grounded_response(s)

            candidate_entry: dict[str, Any] = {
                "sample_id": s.id,
                "group_id": s.source.group_id,
                "source_id": s.source.source_id,
                "source_record_id": s.source.source_record_id,
                "source_file_path": s.source.source_file_path,
                "task_type": s.task_type.value,
                "kep_status": s.source.kep_status,
                "scenario": s.scenario,
                "assistant_response": sft_response,
                "evidence": {
                    "facts": [f.value for f in s.facts],
                    "decisions": [d.value for d in s.decisions],
                    "alternatives": [a.option for a in s.alternatives],
                    "tradeoffs": [t.value for t in s.tradeoffs],
                },
                "review_status": s.review.status.value,
                "review_notes": "Exported for Gold Seed manual review gate V2.",
            }
            candidates.append(candidate_entry)

        output_path = self.export_dir / "gold_review_candidates.jsonl"
        write_jsonl(candidates, output_path)
        return output_path
