"""
Exact Duplicate and N-Gram Jaccard Near-Duplicate Detection Engine
"""

from architectai_dataset_builder.models.canonical import ArchitectAISample


class Deduplicator:
    def __init__(self, jaccard_threshold: float = 0.85, ngram_size: int = 3):
        self.jaccard_threshold = jaccard_threshold
        self.ngram_size = ngram_size

    def _get_ngrams(self, text: str) -> set[str]:
        tokens = text.lower().split()
        if len(tokens) < self.ngram_size:
            return set(tokens)
        return {" ".join(tokens[i : i + self.ngram_size]) for i in range(len(tokens) - self.ngram_size + 1)}

    def compute_jaccard_similarity(self, text_a: str, text_b: str) -> float:
        ngrams_a = self._get_ngrams(text_a)
        ngrams_b = self._get_ngrams(text_b)
        if not ngrams_a or not ngrams_b:
            return 0.0
        intersection = ngrams_a.intersection(ngrams_b)
        union = ngrams_a.union(ngrams_b)
        return len(intersection) / len(union)

    def process_samples(
        self, samples: list[ArchitectAISample]
    ) -> tuple[list[ArchitectAISample], int, int]:
        unique_samples: list[ArchitectAISample] = []
        seen_exact_hashes: set[str] = set()
        seen_texts: list[tuple[str, str]] = []  # (id, text)

        exact_dups = 0
        near_dups = 0

        for sample in samples:
            # 1. Exact Duplicate Hash Match
            norm_hash = sample.source.normalized_sha256
            if norm_hash in seen_exact_hashes:
                exact_dups += 1
                continue

            # 2. Near Duplicate Jaccard Match
            sample_text = f"{sample.scenario} {sample.final_answer or ''}"
            is_near_dup = False
            for _, seen_text in seen_texts:
                sim = self.compute_jaccard_similarity(sample_text, seen_text)
                if sim >= self.jaccard_threshold:
                    is_near_dup = True
                    near_dups += 1
                    break

            if is_near_dup:
                continue

            seen_exact_hashes.add(norm_hash)
            seen_texts.append((sample.id, sample_text))
            unique_samples.append(sample)

        return unique_samples, exact_dups, near_dups
