from __future__ import annotations

import torch

from fish_vlm.evaluation.calibration import fit_calibration


def test_calibration_remaps_subset_head_and_ignores_held_out_targets() -> None:
    dino = torch.tensor(
        [
            [5.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
        ]
    )
    bioclip = dino.clone()
    targets = torch.tensor([0, 2, 1, 2])
    supervised = torch.tensor(
        [
            [5.0, 0.0],
            [1.0, 1.0],
            [0.0, 5.0],
            [1.0, 1.0],
        ]
    )

    calibration = fit_calibration(
        dino,
        bioclip,
        targets,
        supervised,
        supervised_class_indices=[0, 1],
    )

    assert calibration.supervised_temperature > 0
    assert 0 <= calibration.supervised_weight <= 1
