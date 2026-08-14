"""
Protected Evaluation Benchmark Coverage & Diagnostics Reporter
"""

from pathlib import Path
from typing import Any

from architectai_dataset_builder.utils.io import write_jsonl
from architectai_dataset_builder.utils.r2abench_discovery import get_r2abench_discovery_diagnostics


class EvalCoverageReporter:
    """
    Audits protected evaluation benchmarks (SAKE, CAKE, ArchBench, R2ABench)
    and reports discovery diagnostics and held-out coverage.
    """

    def generate_report(
        self,
        eval_samples: list[Any],
        raw_data_dir: Path,
        expected_r2a_heldout_pids: list[str],
        output_file: Path | None = None,
    ) -> dict[str, Any]:
        samples_by_benchmark: dict[str, list[Any]] = {
            "sake": [],
            "cake": [],
            "archbench": [],
            "r2abench": [],
        }

        for s in eval_samples:
            bm_id = s.source.benchmark_id
            if bm_id in samples_by_benchmark:
                samples_by_benchmark[bm_id].append(s)

        r2a_diag = get_r2abench_discovery_diagnostics(
            raw_data_dir / "r2abench", expected_r2a_heldout_pids
        )

        report = {
            "sake": {
                "benchmark_id": "sake",
                "protected": True,
                "evaluation_only": True,
                "license_status": "verified",
                "exported_eval_samples": len(samples_by_benchmark["sake"]),
                "sample_ids": [s.id for s in samples_by_benchmark["sake"]],
            },
            "cake": {
                "benchmark_id": "cake",
                "protected": True,
                "evaluation_only": True,
                "license_status": "verified",
                "exported_eval_samples": len(samples_by_benchmark["cake"]),
                "sample_ids": [s.id for s in samples_by_benchmark["cake"]],
            },
            "archbench": {
                "benchmark_id": "archbench",
                "protected": True,
                "evaluation_only": True,
                "license_status": "verified",
                "exported_eval_samples": len(samples_by_benchmark["archbench"]),
                "sample_ids": [s.id for s in samples_by_benchmark["archbench"]],
            },
            "r2abench": {
                "benchmark_id": "r2abench",
                "protected": True,
                "evaluation_only": True,
                "license_status": "verified",
                "configured_heldout_projects": len(expected_r2a_heldout_pids),
                "discovered_heldout_projects": len(r2a_diag["heldout_found"]),
                "missing_projects": r2a_diag["heldout_missing"],
                "exported_eval_samples": len(samples_by_benchmark["r2abench"]),
                "sample_ids": [s.id for s in samples_by_benchmark["r2abench"]],
                "discovery_diagnostics": r2a_diag,
            },
            "summary": {
                "total_exported_eval_samples": len(eval_samples),
                "eval_sources_count": len(samples_by_benchmark),
            },
        }

        if output_file:
            write_jsonl([report], Path(output_file))

        return report
