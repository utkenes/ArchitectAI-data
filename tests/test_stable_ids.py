import pytest

from architectai_dataset_builder.utils.identity import generate_stable_sample_id
from architectai_dataset_builder.validators.identity_validator import (
    IdentityCollisionError,
    IdentityValidator,
)


def test_stable_id_determinism():
    id1 = generate_stable_sample_id(
        source_id="madr",
        file_path="0001-use-madr.md",
        record_id="0001-use-madr",
    )
    id2 = generate_stable_sample_id(
        source_id="madr",
        file_path="0001-use-madr.md",
        record_id="0001-use-madr",
    )
    assert id1 == id2
    assert id1.startswith("arch_")
    assert len(id1) == 21  # "arch_" + 16 hex chars


def test_stable_id_path_differentiation():
    # foo/README.md vs bar/README.md must produce different sample IDs
    id_foo = generate_stable_sample_id(
        source_id="backstage_adrs",
        file_path="foo/README.md",
        record_id="README",
    )
    id_bar = generate_stable_sample_id(
        source_id="backstage_adrs",
        file_path="bar/README.md",
        record_id="README",
    )
    assert id_foo != id_bar


def test_identity_validator_same_content():
    validator = IdentityValidator()
    validator.register_and_validate(
        sample_id="arch_12345",
        content_hash="hash_a",
        source_id="madr",
        file_path="0001.md",
        record_id="0001",
    )
    # Re-registering exact same content should pass
    validator.register_and_validate(
        sample_id="arch_12345",
        content_hash="hash_a",
        source_id="madr",
        file_path="0001.md",
        record_id="0001",
    )
    assert validator.registered_count == 1


def test_identity_validator_collision_blocks_build():
    validator = IdentityValidator()
    validator.register_and_validate(
        sample_id="arch_collision",
        content_hash="hash_a",
        source_id="madr",
        file_path="foo/0001.md",
        record_id="0001",
    )
    with pytest.raises(IdentityCollisionError) as exc_info:
        validator.register_and_validate(
            sample_id="arch_collision",
            content_hash="hash_b",  # Different content!
            source_id="madr",
            file_path="bar/0001.md",
            record_id="0001",
        )
    assert "Sample ID collision detected" in str(exc_info.value)
