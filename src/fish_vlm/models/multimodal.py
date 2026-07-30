"""Complete DINO/BioCLIP multimodal classifier."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from fish_vlm.models.bioclip import encode_bioclip_images
from fish_vlm.models.fusion import (
    CalibrationParameters,
    apply_seen_class_penalty,
    expanded_supervised_probabilities,
    fuse_seen_probabilities,
    fuse_text_probabilities,
)


@dataclass
class ModelOutput:
    """Branch features and logits returned by one forward pass."""

    dino_features: torch.Tensor
    projected_features: torch.Tensor
    dino_text_logits: torch.Tensor
    bioclip_features: torch.Tensor | None
    bioclip_logits: torch.Tensor | None
    supervised_logits: torch.Tensor | None
    bioclip_original_features: torch.Tensor | None = None
    bioclip_supervised_logits: torch.Tensor | None = None


def normalised_similarity_scores(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Compute ``normalise(image) @ normalise(text).T`` in float32."""
    if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("Image and text embeddings must both be matrices")
    if image_embeddings.shape[-1] != text_embeddings.shape[-1]:
        raise ValueError("Embedding and prototype dimensions differ")
    images = F.normalize(image_embeddings.float(), dim=-1)
    texts = F.normalize(text_embeddings.float(), dim=-1)
    return images @ texts.T


def text_logits(
    embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    scale: torch.Tensor | float,
) -> torch.Tensor:
    """Compute scaled float32 cosine-prototype logits."""
    scores = normalised_similarity_scores(embeddings, prototypes)
    return (
        scale.float() * scores
        if torch.is_tensor(scale)
        else float(scale) * scores
    )


class FishMultimodalModel(nn.Module):
    """Coordinate DINO-text, native BioCLIP and optional supervised branches."""

    def __init__(
        self,
        dino: nn.Module,
        projector: nn.Module,
        logit_scale: nn.Module,
        *,
        bioclip: nn.Module | None = None,
        bioclip_adapter: nn.Module | None = None,
        supervised_head: nn.Module | None = None,
        bioclip_classifier: nn.Module | None = None,
        bioclip_text_space: str = "native",
        bioclip_classifier_space: str = "native",
    ) -> None:
        super().__init__()
        self.dino = dino
        self.projector = projector
        self.logit_scale = logit_scale
        self.bioclip = bioclip
        self.bioclip_adapter = bioclip_adapter
        self.supervised_head = supervised_head
        self.bioclip_classifier = bioclip_classifier
        if bioclip_text_space not in {"native", "adapter"}:
            raise ValueError("bioclip_text_space must be native or adapter")
        if bioclip_classifier_space not in {"native", "adapter"}:
            raise ValueError(
                "bioclip_classifier_space must be native or adapter"
            )
        self.bioclip_text_space = bioclip_text_space
        self.bioclip_classifier_space = bioclip_classifier_space

    def forward(
        self,
        dino_image: torch.Tensor,
        prototypes: torch.Tensor,
        bioclip_image: torch.Tensor | None = None,
    ) -> ModelOutput:
        from fish_vlm.models.dino import pooled_features

        dino_features = pooled_features(self.dino, dino_image)
        projected = self.projector(dino_features)
        dino_logits = text_logits(projected, prototypes, self.logit_scale())
        bioclip_features = None
        bioclip_original_features = None
        bioclip_logits = None
        bioclip_supervised_logits = None
        if self.bioclip is not None and bioclip_image is not None:
            bioclip_original_features = encode_bioclip_images(
                self.bioclip, bioclip_image
            )
            adapted_features = bioclip_original_features
            if self.bioclip_adapter is not None:
                adapted_features = self.bioclip_adapter(
                    bioclip_original_features
                )
            bioclip_features = (
                adapted_features
                if self.bioclip_text_space == "adapter"
                else bioclip_original_features
            )
            bioclip_logits = text_logits(bioclip_features, prototypes, self.logit_scale())
            if self.bioclip_classifier is not None:
                classifier_features = (
                    adapted_features
                    if self.bioclip_classifier_space == "adapter"
                    else bioclip_original_features
                )
                bioclip_supervised_logits = self.bioclip_classifier(
                    classifier_features
                )
        supervised_logits = self.supervised_head(dino_features) if self.supervised_head is not None else None
        return ModelOutput(
            dino_features,
            projected,
            dino_logits,
            bioclip_features,
            bioclip_logits,
            supervised_logits,
            bioclip_original_features,
            bioclip_supervised_logits,
        )

    @staticmethod
    def probabilities(
        output: ModelOutput,
        mode: str,
        calibration: CalibrationParameters,
        *,
        supervised_class_indices: list[int] | torch.Tensor | None = None,
        seen_class_indices: list[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return probabilities for an explicit inference mode."""
        if mode == "dino_text":
            return torch.softmax(output.dino_text_logits.float() / calibration.dino_temperature, dim=-1)
        if mode == "bioclip_native":
            if output.bioclip_logits is None:
                raise ValueError("BioCLIP-native branch is unavailable")
            return torch.softmax(output.bioclip_logits.float() / calibration.bioclip_temperature, dim=-1)
        if mode == "bioclip_supervised":
            if output.bioclip_supervised_logits is None:
                raise ValueError("BioCLIP supervised classifier is unavailable")
            return expanded_supervised_probabilities(
                output.bioclip_supervised_logits,
                calibration.supervised_temperature,
                class_count=output.dino_text_logits.shape[1],
                class_indices=supervised_class_indices,
            )
        if mode == "bioclip_supervised_plus_text":
            if (
                output.bioclip_supervised_logits is None
                or output.bioclip_logits is None
            ):
                raise ValueError(
                    "BioCLIP seen fusion requires classifier and text logits"
                )
            text = torch.softmax(
                output.bioclip_logits.float()
                / calibration.bioclip_temperature,
                dim=-1,
            )
            return fuse_seen_probabilities(
                output.bioclip_supervised_logits,
                text,
                calibration,
                supervised_class_indices=supervised_class_indices,
            )
        if mode == "fused_text":
            if output.bioclip_logits is None:
                raise ValueError("BioCLIP-native branch is unavailable")
            probabilities = fuse_text_probabilities(
                output.dino_text_logits,
                output.bioclip_logits,
                calibration,
            )
            if seen_class_indices is not None:
                probabilities = apply_seen_class_penalty(
                    probabilities,
                    seen_class_indices,
                    calibration.calibration_gamma,
                )
            return probabilities
        if mode == "supervised":
            if output.supervised_logits is None:
                raise ValueError("Supervised branch is unavailable")
            return expanded_supervised_probabilities(
                output.supervised_logits,
                calibration.supervised_temperature,
                class_count=output.dino_text_logits.shape[1],
                class_indices=supervised_class_indices,
            )
        if mode == "supervised_plus_text":
            if output.supervised_logits is None or output.bioclip_logits is None:
                raise ValueError("Seen fusion requires supervised and both text branches")
            text = fuse_text_probabilities(output.dino_text_logits, output.bioclip_logits, calibration)
            return fuse_seen_probabilities(
                output.supervised_logits,
                text,
                calibration,
                supervised_class_indices=supervised_class_indices,
            )
        raise ValueError(f"Unknown inference mode: {mode}")
