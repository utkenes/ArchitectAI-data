from pathlib import Path

from architectai_dataset_builder.models.canonical import (
    ArchitectAISample,
    ReviewInfo,
    ReviewStatus,
    SourceMetadata,
    TaskType,
)
from architectai_dataset_builder.splitters.deterministic_splitter import DeterministicSplitter


def test_group_level_sibling_isolation(tmp_path: Path):
    manifest_path = tmp_path / "splits.yaml"
    manifest_path.write_text("splits:\n  training_projects: []\n", encoding="utf-8")

    splitter = DeterministicSplitter(manifest_path)

    samples = []
    # Create 5 sibling samples sharing group_id='group_k8s_keps_sig-arch_100'
    for i in range(5):
        s = ArchitectAISample(
            id=f"sample_{i}",
            source=SourceMetadata(
                source_id="k8s_keps",
                source_name="K8s KEPs",
                source_file_path="kep.md",
                source_record_id="100",
                project_id="sig-arch",
                group_id="group_k8s_keps_sig-arch_100",
                license_id="Apache-2.0",
                license_verified=True,
                raw_sha256="abc",
                normalized_sha256="def",
            ),
            scenario="Scenario test",
            task_type=TaskType.ADR_REASONING,
            review=ReviewInfo(status=ReviewStatus.UNREVIEWED),
        )
        samples.append(s)

    train, val = splitter.split_samples(samples)

    # Verify that all 5 samples landed in the exact same split (either 100% train or 100% val)
    assert len(train) == 5 or len(val) == 5
