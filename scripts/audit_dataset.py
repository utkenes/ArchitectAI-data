"""
Comprehensive Audit Script for ArchitectAI Dataset Builder V1.1
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from architectai_dataset_builder.config import Config
from architectai_dataset_builder.utils.io import read_jsonl


def main():
    cfg = Config()
    export_dir = cfg.data_dir / "exports"

    manifest_file = export_dir / "build_manifest.json"
    stats_file = export_dir / "dataset_stats.json"
    readiness_file = export_dir / "training_readiness_report.json"
    contamination_file = export_dir / "contamination_report.json"
    train_sft_file = export_dir / "train_sft.jsonl"
    val_sft_file = export_dir / "validation_sft.jsonl"
    gold_candidates_file = export_dir / "gold_review_candidates.jsonl"

    if not manifest_file.exists():
        print(f"Error: Build manifest not found at {manifest_file}. Run build script first.")
        sys.exit(1)

    manifest_data = read_jsonl(manifest_file)[0]
    stats_data = read_jsonl(stats_file)[0]
    readiness_data = read_jsonl(readiness_file)[0] if readiness_file.exists() else {}
    contamination_data = read_jsonl(contamination_file)[0] if contamination_file.exists() else {}

    gold_candidate_count = len(read_jsonl(gold_candidates_file)) if gold_candidates_file.exists() else 0

    print("\n========================================================")
    print("      ArchitectAI V1.1 Post-Build Mandatory Audit       ")
    print("========================================================")
    print(f"1.  Build ID:                  {manifest_data.get('build_id')}")
    print(f"2.  Total Samples (Silver):    {stats_data.get('total_samples')}")
    print(f"3.  Train Count:               {readiness_data.get('train_samples')}")
    print(f"4.  Validation Count:          {readiness_data.get('validation_samples')}")
    print(f"5.  Source Distribution:       {readiness_data.get('source_distribution')}")
    conc_ratio = readiness_data.get('source_concentration_ratio', 0)
    print(f"6.  Source Concentration %:    {conc_ratio * 100:.2f}%")
    print(f"7.  Task Distribution:         {readiness_data.get('task_distribution')}")
    print(f"8.  Duplicate Sample IDs:      {readiness_data.get('duplicate_sample_ids')}")
    print(f"9.  Exact Duplicate Content:   {stats_data.get('exact_duplicates_filtered')}")
    print(f"10. Near Duplicates:           {stats_data.get('near_duplicates_filtered')}")
    print(f"11. Template Leakage Count:    {readiness_data.get('template_leakage_count')}")
    print(f"12. Quarantine Breakdown:      {stats_data.get('quarantine_reason_breakdown')}")
    print(f"13. Group Leakage:             {readiness_data.get('group_split_integrity')}")
    print(f"14. Eval Contamination:        {readiness_data.get('train_eval_contamination')}")
    print(f"15. Gold Candidate Count:      {gold_candidate_count}")
    print(f"16. Reviewed Gold Count:       {readiness_data.get('gold_samples')}")
    print(f"17. Eval Benchmark Counts:     {readiness_data.get('evaluation_counts')}")
    print(f"18. Readiness Status:          {readiness_data.get('readiness_status')}")

    # SFT Assistant Response Token Leakage Scan
    sft_files = [f for f in [train_sft_file, val_sft_file] if f.exists()]
    leakage_findings = []
    suspicious_todos = []
    empty_answers = 0

    for sft_f in sft_files:
        samples = read_jsonl(sft_f)
        for s in samples:
            messages = s.get("messages", [])
            assistant_text = ""
            for m in messages:
                if m.get("role") == "assistant":
                    assistant_text = m.get("content", "")

            if not assistant_text.strip():
                empty_answers += 1

            for token in ["<!--", "-->", "{{", "${"]:
                if token in assistant_text:
                    leakage_findings.append(f"Sample {s.get('id')}: Found raw token '{token}' in assistant response.")

            if "TODO" in assistant_text:
                suspicious_todos.append(f"Sample {s.get('id')}: Contains TODO in response.")
            if "TBD" in assistant_text:
                suspicious_todos.append(f"Sample {s.get('id')}: Contains TBD in response.")

    print("\n--- SFT Response Token Scan ---")
    print(f"Empty Assistant Answers:       {empty_answers}")
    print(f"Raw Token Leakage Findings:    {len(leakage_findings)}")
    print(f"Suspicious TODO/TBD Findings:  {len(suspicious_todos)}")

    if leakage_findings:
        print("CRITICAL: Raw token leakages detected in SFT dataset:")
        for lf in leakage_findings[:10]:
            print(f"  - {lf}")

    if suspicious_todos:
        print("NOTE: Inspect contextual TODO/TBD findings:")
        for st in suspicious_todos[:5]:
            print(f"  - {st}")

    if contamination_data.get("has_leakage") or readiness_data.get("readiness_status") == "BUILD_INVALID":
        print("\nCRITICAL ERROR: Build is INVALID or has contamination!")
        sys.exit(1)
    else:
        print("\nAudit completed cleanly.")


if __name__ == "__main__":
    main()
