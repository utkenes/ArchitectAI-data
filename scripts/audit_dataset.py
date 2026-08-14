"""
Comprehensive Audit Script for ArchitectAI Dataset Builder V1.1.1
Consumes SFTExportValidator as shared source of truth.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from architectai_dataset_builder.config import Config
from architectai_dataset_builder.utils.io import read_jsonl
from architectai_dataset_builder.validators.sft_export_validator import SFTExportValidator


def main():
    cfg = Config()
    export_dir = cfg.data_dir / "exports"

    manifest_file = export_dir / "build_manifest.json"
    stats_file = export_dir / "dataset_stats.json"
    readiness_file = export_dir / "training_readiness_report.json"
    contamination_file = export_dir / "contamination_report.json"
    gold_candidates_file = export_dir / "gold_review_candidates.jsonl"

    if not manifest_file.exists():
        print(f"Error: Build manifest not found at {manifest_file}. Run build script first.")
        sys.exit(1)

    manifest_data = read_jsonl(manifest_file)[0]
    stats_data = read_jsonl(stats_file)[0]
    readiness_data = read_jsonl(readiness_file)[0] if readiness_file.exists() else {}
    contamination_data = read_jsonl(contamination_file)[0] if contamination_file.exists() else {}

    gold_candidate_count = len(read_jsonl(gold_candidates_file)) if gold_candidates_file.exists() else 0

    # Execute shared SFTExportValidator
    export_val = SFTExportValidator().validate_exports(export_dir)

    print("\n========================================================")
    print("     ArchitectAI V1.2 Post-Build Mandatory Audit        ")
    print("========================================================")
    print(f"1.  Build ID:                        {manifest_data.get('build_id')}")
    print(f"2.  Total Silver Samples:            {stats_data.get('total_samples')}")
    print(f"3.  Train Count:                     {readiness_data.get('train_samples')}")
    print(f"4.  Validation Count:                {readiness_data.get('validation_samples')}")
    print(f"5.  Source Distribution:             {readiness_data.get('source_distribution')}")
    conc_ratio = readiness_data.get('source_concentration_ratio', 0)
    print(f"6.  Source Concentration %:          {conc_ratio * 100:.2f}%")
    print(f"7.  Task Distribution:               {readiness_data.get('task_distribution')}")
    print(f"8.  Duplicate Sample IDs:            {export_val.duplicate_sample_ids}")
    print(f"9.  Conflicting Duplicate IDs:       {export_val.conflicting_duplicate_sample_ids}")
    print(f"10. Exact Duplicate Content:         {stats_data.get('exact_duplicate_count')}")
    print(f"11. Near Duplicates:                 {stats_data.get('near_duplicate_count')}")
    print(f"12. Template Leakage Count:          {export_val.template_leakage_count}")
    print(f"13. Unresolved Placeholders:         {export_val.unresolved_placeholder_count}")
    print(f"14. Empty Assistant Answers:         {export_val.empty_assistant_answers}")
    print(f"15. Quarantine Breakdown:            {stats_data.get('quarantine_reasons')}")
    print(f"16. Group Leakage:                   {readiness_data.get('group_split_integrity')}")
    print(f"17. Eval Contamination:              {readiness_data.get('train_eval_contamination')}")
    print(f"18. Gold Candidates Count:           {gold_candidate_count}")
    print(f"19. Reviewed Gold Count:             {readiness_data.get('gold_samples')}")
    print(f"20. Manual Review Completed:         {readiness_data.get('manual_review_completed')}")
    print(f"21. Eval Benchmark Counts:           {readiness_data.get('evaluation_counts')}")
    print(f"22. Readiness Status:                {readiness_data.get('readiness_status')}")

    if readiness_data.get("blocking_reasons"):
        print("\n--- BLOCKING REASONS ---")
        for br in readiness_data.get("blocking_reasons", []):
            print(f"  - {br}")

    if readiness_data.get("warnings"):
        print("\n--- READINESS WARNINGS ---")
        for w in readiness_data.get("warnings", []):
            print(f"  - {w}")

    print("\n--- SFT Export Validator Inspection ---")
    print(f"Raw Token Leakage Findings:          {len(export_val.raw_token_leakage_findings)}")
    print(f"Suspicious TODO/TBD Findings:        {len(export_val.suspicious_todos)}")

    if export_val.raw_token_leakage_findings:
        print("CRITICAL: Raw token leakages detected in SFT dataset:")
        for lf in export_val.raw_token_leakage_findings[:10]:
            print(f"  - {lf}")

    if export_val.suspicious_todos:
        print("NOTE: Inspect contextual TODO/TBD findings:")
        for st in export_val.suspicious_todos[:5]:
            print(f"  - {st}")

    if (
        contamination_data.get("has_leakage")
        or readiness_data.get("readiness_status") == "BUILD_INVALID"
        or export_val.has_critical_failures
    ):
        print("\nCRITICAL ERROR: Build is INVALID or has contamination / export failures!")
        sys.exit(1)
    else:
        print("\nAudit completed cleanly.")


if __name__ == "__main__":
    main()
