"""
Contamination Checker & Evaluation Leakage Enforcement Engine
"""


from architectai_dataset_builder.dedup.deduplicator import Deduplicator
from architectai_dataset_builder.models.canonical import ArchitectAISample
from architectai_dataset_builder.models.evaluation import (
    ArchitectureGenerationEvalSample,
    DiagramEvalSample,
    FreeResponseEvalSample,
    MultipleChoiceEvalSample,
)
from architectai_dataset_builder.models.reports import ContaminationReport, LeakageDetail

EvalSampleUnion = (
    MultipleChoiceEvalSample
    | FreeResponseEvalSample
    | ArchitectureGenerationEvalSample
    | DiagramEvalSample
)


class EvaluationLeakageError(Exception):
    """Raised when an evaluation benchmark sample leaks into a training split."""



class ContaminationChecker:
    def __init__(self, protected_sources: list[str], jaccard_threshold: float = 0.85):
        self.protected_sources = set(protected_sources)
        self.deduplicator = Deduplicator(jaccard_threshold=jaccard_threshold)

    def check_contamination(
        self,
        training_samples: list[ArchitectAISample],
        eval_samples: list[EvalSampleUnion],
    ) -> ContaminationReport:
        leakage_details: list[LeakageDetail] = []

        # Index evaluation samples
        eval_raw_hashes: set[str] = set()
        eval_norm_hashes: set[str] = set()
        eval_records: list[tuple[str, str, str]] = []  # (sample_id, source_id, text)

        for es in eval_samples:
            eval_raw_hashes.add(es.source.raw_sha256)
            eval_norm_hashes.add(es.source.normalized_sha256)

            # Extract text
            if isinstance(es, MultipleChoiceEvalSample):
                text = f"{es.question} {' '.join(es.options.values())}"
            elif isinstance(es, FreeResponseEvalSample):
                text = f"{es.prompt} {es.reference_answer}"
            elif isinstance(es, ArchitectureGenerationEvalSample):
                text = f"{es.requirements} {es.reference_solution}"
            elif isinstance(es, DiagramEvalSample):
                text = f"{es.requirements_text} {es.reference_plantuml}"
            else:
                text = ""

            eval_records.append((es.id, es.source.benchmark_id, text))

        # Check each training sample against evaluation index
        for ts in training_samples:
            # 1. Metadata-level source check
            if ts.source.source_id in self.protected_sources:
                leakage_details.append(
                    LeakageDetail(
                        training_sample_id=ts.id,
                        evaluation_sample_id="source_level_leak",
                        match_type="protected_source_in_training",
                        similarity_score=1.0,
                        source_a=ts.source.source_id,
                        source_b=ts.source.source_id,
                    )
                )

            # 2. Content Hash Matches
            if ts.source.raw_sha256 in eval_raw_hashes:
                leakage_details.append(
                    LeakageDetail(
                        training_sample_id=ts.id,
                        evaluation_sample_id="raw_hash_match",
                        match_type="raw_sha256_exact_match",
                        similarity_score=1.0,
                        source_a=ts.source.source_id,
                        source_b="eval_benchmark",
                    )
                )

            if ts.source.normalized_sha256 in eval_norm_hashes:
                leakage_details.append(
                    LeakageDetail(
                        training_sample_id=ts.id,
                        evaluation_sample_id="norm_hash_match",
                        match_type="normalized_sha256_exact_match",
                        similarity_score=1.0,
                        source_a=ts.source.source_id,
                        source_b="eval_benchmark",
                    )
                )

            # 3. Near-Duplicate Text Similarity
            ts_text = f"{ts.scenario} {ts.final_answer or ''}"
            for eval_id, eval_source_id, eval_text in eval_records:
                sim = self.deduplicator.compute_jaccard_similarity(ts_text, eval_text)
                if sim >= self.deduplicator.jaccard_threshold:
                    leakage_details.append(
                        LeakageDetail(
                            training_sample_id=ts.id,
                            evaluation_sample_id=eval_id,
                            match_type="ngram_jaccard_similarity",
                            similarity_score=round(sim, 3),
                            source_a=ts.source.source_id,
                            source_b=eval_source_id,
                        )
                    )

        report = ContaminationReport(
            has_leakage=len(leakage_details) > 0,
            total_leaks_detected=len(leakage_details),
            leakage_details=leakage_details,
        )

        return report

    def verify_no_leakage(
        self,
        training_samples: list[ArchitectAISample],
        eval_samples: list[EvalSampleUnion],
    ) -> ContaminationReport:
        report = self.check_contamination(training_samples, eval_samples)
        if report.has_leakage:
            raise EvaluationLeakageError(
                f"Evaluation Contamination Error! Detected {report.total_leaks_detected} leaks between training and evaluation."
            )
        return report
