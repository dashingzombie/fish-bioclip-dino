import pytest
import torch

from conftest import TinyBioClip
from fish_vlm.inference.predict import decode_class_scores
from fish_vlm.models.bioclip import encode_bioclip_images
from fish_vlm.models.multimodal import (
    normalised_similarity_scores,
    text_logits,
)


def test_candidate_prototypes_restrict_logits() -> None:
    embeddings = torch.nn.functional.normalize(torch.randn(4, 3), dim=-1)
    seen = torch.nn.functional.normalize(torch.randn(2, 3), dim=-1)
    unseen = torch.nn.functional.normalize(torch.randn(5, 3), dim=-1)
    assert text_logits(embeddings, seen, 10.0).shape == (4, 2)
    assert text_logits(embeddings, unseen, 10.0).shape == (4, 5)
    with pytest.raises(ValueError, match="dimensions"):
        text_logits(embeddings, torch.randn(2, 4), 1.0)


def test_native_bioclip_logits_use_the_same_prototypes() -> None:
    model = TinyBioClip()
    images = torch.eye(4)[:2]
    prototypes = torch.nn.functional.normalize(torch.randn(3, 3), dim=-1)
    embedding = encode_bioclip_images(model, images)
    logits = text_logits(embedding, prototypes, 5.0)
    assert embedding.shape == (2, 3)
    assert logits.shape == (2, 3)


def test_normalised_scores_and_decoding_share_exact_class_order() -> None:
    image_embeddings = torch.tensor([[10.0, 0.0], [0.0, 2.0]])
    text_embeddings = torch.tensor(
        [
            [0.0, 3.0],  # column 0: B fish
            [4.0, 0.0],  # column 1: A fish
        ]
    )
    class_names = ["B fish", "A fish"]

    scores = normalised_similarity_scores(
        image_embeddings, text_embeddings
    )

    assert torch.equal(scores, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    assert decode_class_scores(scores, class_names) == ["A fish", "B fish"]
    with pytest.raises(ValueError, match="column count"):
        decode_class_scores(scores, ["A fish"])


def test_text_logits_normalise_both_embedding_sides() -> None:
    images = torch.tensor([[100.0, 0.0]])
    texts = torch.tensor([[2.0, 0.0], [0.0, 0.5]])
    assert torch.equal(
        text_logits(images, texts, 5.0),
        torch.tensor([[5.0, 0.0]]),
    )
