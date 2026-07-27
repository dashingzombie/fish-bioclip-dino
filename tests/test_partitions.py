from fish_vlm.data.partitions import build_class_partitions


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

