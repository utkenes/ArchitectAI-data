from architectai_dataset_builder.utils.identity import generate_stable_sample_id


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


def test_stable_id_uniqueness():
    id1 = generate_stable_sample_id("madr", "0001.md", "rec1")
    id2 = generate_stable_sample_id("madr", "0002.md", "rec2")
    assert id1 != id2
