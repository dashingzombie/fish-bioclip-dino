import pytest

from fish_vlm.data.partitions import ClassPartitions, build_class_partitions


def test_partitions_are_sorted_and_mapped() -> None:
    partitions = build_class_partitions(
        {"b.jpg": "Zeta beta", "a.jpg": "Alpha fish"},
        ["Unseen one", "Zeta beta", "Alpha fish"],
    )
    assert partitions.seen_species == ["Alpha fish", "Zeta beta"]
    assert partitions.unseen_species == ["Unseen one"]
    value = partitions.to_dict()
    assert value["seen_species_to_index"] == {"Alpha fish": 0, "Zeta beta": 1}
    assert value["index_to_all_species"]["0"] == "Alpha fish"


def test_persisted_partitions_reject_stale_ordering_mappings() -> None:
    value = build_class_partitions(
        {"a.jpg": "Seen fish"},
        ["Seen fish", "Unseen fish"],
    ).to_dict()
    value["unseen_species_to_index"] = {"Unseen fish": 1}
    with pytest.raises(ValueError, match="mappings"):
        ClassPartitions.from_dict(value)


def test_partitions_reject_seen_unseen_overlap() -> None:
    value = build_class_partitions(
        {"a.jpg": "Seen fish"},
        ["Seen fish", "Unseen fish"],
    ).to_dict()
    value["unseen_species"] = ["Seen fish", "Unseen fish"]
    with pytest.raises(ValueError, match="overlap"):
        ClassPartitions.from_dict(value)
