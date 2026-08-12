# ArchitectAI Dataset Builder V1

A standalone, production-grade, reproducible, license-aware, and contamination-safe training dataset pipeline for ArchitectAI.

## Overview

ArchitectAI Dataset Builder processes raw architectural sources (e.g. OpenDataHub ADRs, MADR templates/examples, R2ABench requirements) and builds clean, provenance-traceable SFT and evaluation datasets.

### Strict Data Class Separation
The pipeline enforces four distinct data classes:
- **GOLD**: Human-approved architecture samples (`manifests/reviews/approved_samples.yaml`).
- **SILVER**: Automated schema-valid, license-verified, deduplicated, relevance-filtered training candidates.
- **PREFERENCE / NEGATIVE**: Reserved models for future DPO/ORPO preference datasets (V2).
- **HELD-OUT EVALUATION**: Benchmark datasets (`SAKE`, `CAKE`, `ArchBench`, `R2ABench Holdout`) strictly isolated from training export.

### Primary Rule: Zero Train/Eval Contamination
> Evaluation datasets must never leak into training datasets.

Any detected overlap across metadata or content levels will raise an explicit `EvaluationLeakageError` and block dataset export.

## Pipeline Workflow

```text
RAW SOURCE (Immutable, SHA-256 Verified)
  ↓
PARSER & NORMALIZER (Canonical Training vs Eval Schema Isolation)
  ↓
VALIDATOR (Schema & Pinned License Gating)
  ↓
RELEVANCE FILTER & DEDUPLICATOR (Exact & N-Gram Jaccard)
  ↓
SPLITTER & CONTAMINATION CHECKER (Zero Eval Leakage Enforcement)
  ↓
EXPORTER (JSONL, SFT Messages & Build Manifest)
```

## Quick Start & Verification Commands

```bash
# Install package in editable mode
pip install -e .[dev]

# Run unit tests, linting, and type checks
pytest -v
ruff check src tests
mypy src

# Execute build pipeline
python scripts/bootstrap_sources.py
python scripts/build_dataset.py
python scripts/audit_dataset.py

# CLI Commands
architectai-data sources list
architectai-data sources verify
architectai-data build
```
