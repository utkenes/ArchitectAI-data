"""
Source Manifest, License Policy, Review Manifest, and Build Manifest Models
"""

from typing import List, Dict, Any, Optional
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
    integrity: Dict[str, Optional[str]] = Field(default_factory=lambda: {"raw_sha256": None})
    notes: Optional[str] = None


class ApprovedSampleEntry(BaseModel):
    sample_id: str
    reviewer: str
    approved_at: str
    notes: Optional[str] = None


class ReviewManifest(BaseModel):
    approved_samples: List[ApprovedSampleEntry] = Field(default_factory=list)


class BuildManifest(BaseModel):
    dataset_version: str
    builder_version: str
    build_id: str
    build_timestamp: str
    config_hash: str
    split_manifest_hash: str
    sources: Dict[str, Dict[str, Any]]
    sample_counts: Dict[str, int]
    export_hashes: Dict[str, str]
