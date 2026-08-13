"""
Source Manifest, License Policy, Review Manifest, and Build Manifest Models
"""

from typing import Any, Optional
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
    verification_source: Optional[str] = None
    policy: LicensePolicy = Field(default_factory=LicensePolicy)


class SourceOrigin(BaseModel):
    provider: str
    repository_url: Optional[str] = None


class SourceVersion(BaseModel):
    revision: Optional[str] = None
    commit_sha: Optional[str] = None
    tag: Optional[str] = None
    release_version: Optional[str] = None
    retrieved_at: Optional[str] = None
    requested_ref: Optional[str] = None
    resolved_commit: Optional[str] = None


class SourcePolicy(BaseModel):
    training_allowed: bool = True
    evaluation_only: bool = False
    redistribution_allowed: Optional[bool] = None


class SourceManifest(BaseModel):
    source_id: str
    name: str
    source_type: str
    origin: SourceOrigin
    version: SourceVersion
    license: LicenseMetadata
    policy: SourcePolicy
    integrity: dict[str, Optional[str]] = Field(default_factory=dict)
    notes: Optional[str] = None


class ApprovedSampleEntry(BaseModel):
    sample_id: str
    reviewer: str
    approved_at: str
    notes: Optional[str] = None


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
