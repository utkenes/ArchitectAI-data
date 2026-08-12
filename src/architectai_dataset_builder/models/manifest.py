"""
Source Manifest, License Policy, Review Manifest, and Build Manifest Models
"""

from typing import Any

from pydantic import BaseModel, Field


class LicensePolicy(BaseModel):
    ingestion_allowed: bool = True
    training_allowed: bool = True
    redistribution_allowed: bool = True
    modification_allowed: bool = True
    commercial_use_allowed: bool = True
    attribution_required: bool = True


class LicenseMetadata(BaseModel):
    spdx_id: str
    verified: bool = False
    verification_source: str | None = None
    policy: LicensePolicy = Field(default_factory=LicensePolicy)


class SourceOrigin(BaseModel):
    provider: str
    repository_url: str | None = None


class SourceVersion(BaseModel):
    revision: str | None = None
    commit_sha: str | None = None
    tag: str | None = None
    release_version: str | None = None
    retrieved_at: str | None = None


class SourcePolicy(BaseModel):
    training_allowed: bool = True
    evaluation_only: bool = False
    redistribution_allowed: bool | None = None


class SourceManifest(BaseModel):
    source_id: str
    name: str
    source_type: str
    origin: SourceOrigin
    version: SourceVersion
    license: LicenseMetadata
    policy: SourcePolicy
    integrity: dict[str, str | None] = Field(default_factory=dict)
    notes: str | None = None


class ApprovedSampleEntry(BaseModel):
    sample_id: str
    reviewer: str
    approved_at: str
    notes: str | None = None


class ReviewManifest(BaseModel):
    approved_samples: list[ApprovedSampleEntry] = Field(default_factory=list)


class BuildManifest(BaseModel):
    dataset_version: str
    builder_version: str
    build_id: str
    build_timestamp: str
    config_hash: str
    split_manifest_hash: str
    sources: dict[str, dict[str, Any]]
    sample_counts: dict[str, int]
    export_hashes: dict[str, str]
