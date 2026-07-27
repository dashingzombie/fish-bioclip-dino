import torch

from fish_vlm.models.projector import (
    DinoToBioClipProjector,
    LearnableLogitScale,
    LinearDinoToBioClipProjector,
)


def test_projector_shapes_and_normalisation() -> None:
    inputs = torch.randn(5, 7)
    for projector in (DinoToBioClipProjector(7, 3, 11), LinearDinoToBioClipProjector(7, 3)):
        output = projector(inputs)
        assert output.shape == (5, 3)
        assert output.dtype == torch.float32
        assert torch.allclose(output.norm(dim=-1), torch.ones(5), atol=1e-5)
    assert 0 < LearnableLogitScale()().item() <= 100

