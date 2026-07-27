"""Persistent deterministic DINO/BioCLIP image-transform caches."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from fish_vlm.utils.io import read_json, write_json


CACHE_VERSION = 1


def validate_image_filenames(filenames: list[str]) -> list[str]:
    """Reject duplicate, absolute, or escaping paths before staging/caching."""
    names = list(filenames)
    if len(names) != len(set(names)):
        raise ValueError("Image-cache filenames must be unique")
    for name in names:
        path = PurePosixPath(name)
        if not name or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe image filename: {name!r}")
    return names


class _RawTransformDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        filenames: list[str],
        images_dir: Path,
        dino_transform: Any,
        bioclip_transform: Any,
    ) -> None:
        self.filenames = filenames
        self.images_dir = images_dir
        self.dino_transform = dino_transform
        self.bioclip_transform = bioclip_transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        filename = self.filenames[index]
        with Image.open(self.images_dir / filename) as source:
            image = source.convert("RGB")
            return {
                "filename": filename,
                "dino_image": self.dino_transform(image),
                "bioclip_image": self.bioclip_transform(image),
            }


def _collate_transforms(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "filename": [item["filename"] for item in batch],
        "dino_image": torch.stack([item["dino_image"] for item in batch]),
        "bioclip_image": torch.stack([item["bioclip_image"] for item in batch]),
    }


@dataclass
class DeterministicImageCache:
    """One split's memory-mapped deterministic model inputs."""

    path: Path
    manifest: dict[str, Any]
    _dino: np.ndarray | None = field(default=None, init=False, repr=False)
    _bioclip: np.ndarray | None = field(default=None, init=False, repr=False)

    @property
    def filenames(self) -> list[str]:
        return list(self.manifest["filenames"])

    def _open(self) -> None:
        if self._dino is None:
            self._dino = np.load(self.path / "dino.npy", mmap_mode="r")
            self._bioclip = np.load(self.path / "bioclip.npy", mmap_mode="r")

    def get(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._open()
        assert self._dino is not None and self._bioclip is not None
        dino = torch.from_numpy(np.array(self._dino[index], copy=True)).float()
        bioclip = torch.from_numpy(np.array(self._bioclip[index], copy=True)).float()
        return dino, bioclip


class DeterministicImageCacheCollection:
    """Resolve filenames across one or more disjoint split caches."""

    def __init__(self, caches: list[DeterministicImageCache]) -> None:
        self.caches = caches
        self._locations: dict[str, tuple[int, int]] = {}
        for cache_index, cache in enumerate(caches):
            for item_index, filename in enumerate(cache.filenames):
                if filename in self._locations:
                    raise ValueError(
                        f"Image appears in multiple deterministic caches: {filename}"
                    )
                self._locations[filename] = (cache_index, item_index)

    def require(self, filenames: list[str]) -> None:
        missing = [name for name in filenames if name not in self._locations]
        if missing:
            raise FileNotFoundError(
                f"Deterministic image cache misses {len(missing)} files; "
                f"first missing: {missing[:5]}"
            )

    def get(self, filename: str) -> tuple[torch.Tensor, torch.Tensor]:
        cache_index, item_index = self._locations[filename]
        return self.caches[cache_index].get(item_index)


def load_deterministic_image_cache(
    path: str | Path,
    *,
    expected_filenames: list[str],
    dino_model_name: str,
    bioclip_checkpoint: str,
    dino_transform_hash: str | None = None,
    bioclip_transform_hash: str | None = None,
) -> DeterministicImageCache:
    """Validate scientific identity and array shapes before reuse."""
    root = Path(path)
    manifest = read_json(root / "manifest.json")
    expected: dict[str, Any] = {
        "version": CACHE_VERSION,
        "filenames": validate_image_filenames(expected_filenames),
        "dino_model_name": dino_model_name,
        "bioclip_checkpoint": bioclip_checkpoint,
    }
    if dino_transform_hash is not None:
        expected["dino_transform_hash"] = dino_transform_hash
    if bioclip_transform_hash is not None:
        expected["bioclip_transform_hash"] = bioclip_transform_hash
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Incompatible deterministic image cache: {mismatches}")
    if manifest.get("dtype") not in {"float16", "float32"}:
        raise ValueError("Deterministic image cache dtype is invalid")
    dino = np.load(root / "dino.npy", mmap_mode="r")
    bioclip = np.load(root / "bioclip.npy", mmap_mode="r")
    if list(dino.shape) != manifest.get("dino_shape"):
        raise ValueError("Deterministic DINO cache shape is invalid")
    if list(bioclip.shape) != manifest.get("bioclip_shape"):
        raise ValueError("Deterministic BioCLIP cache shape is invalid")
    if dino.shape[0] != len(expected_filenames) or bioclip.shape[0] != len(
        expected_filenames
    ):
        raise ValueError("Deterministic image cache row count is invalid")
    return DeterministicImageCache(root, manifest)


def build_deterministic_image_cache(
    *,
    path: str | Path,
    filenames: list[str],
    images_dir: str | Path,
    dino_transform: Any,
    bioclip_transform: Any,
    dino_model_name: str,
    bioclip_checkpoint: str,
    dino_transform_hash: str,
    bioclip_transform_hash: str,
    dtype: str = "float16",
    batch_size: int = 128,
    num_workers: int = 16,
) -> DeterministicImageCache:
    """Materialize deterministic transforms into two contiguous NPY arrays."""
    destination = Path(path)
    names = validate_image_filenames(filenames)
    if destination.exists():
        return load_deterministic_image_cache(
            destination,
            expected_filenames=names,
            dino_model_name=dino_model_name,
            bioclip_checkpoint=bioclip_checkpoint,
            dino_transform_hash=dino_transform_hash,
            bioclip_transform_hash=bioclip_transform_hash,
        )
    if not names:
        raise ValueError("Cannot build an empty deterministic image cache")
    if dtype not in {"float16", "float32"}:
        raise ValueError("Image cache dtype must be float16 or float32")

    dataset = _RawTransformDataset(
        names,
        Path(images_dir),
        dino_transform,
        bioclip_transform,
    )
    first = dataset[0]
    dino_shape = tuple(first["dino_image"].shape)
    bioclip_shape = tuple(first["bioclip_image"].shape)
    numpy_dtype = np.float16 if dtype == "float16" else np.float32
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        dino_array = np.lib.format.open_memmap(
            temporary / "dino.npy",
            mode="w+",
            dtype=numpy_dtype,
            shape=(len(names), *dino_shape),
        )
        bioclip_array = np.lib.format.open_memmap(
            temporary / "bioclip.npy",
            mode="w+",
            dtype=numpy_dtype,
            shape=(len(names), *bioclip_shape),
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_transforms,
        )
        offset = 0
        observed: list[str] = []
        for batch in loader:
            count = len(batch["filename"])
            observed.extend(batch["filename"])
            dino_array[offset : offset + count] = (
                batch["dino_image"].numpy().astype(numpy_dtype, copy=False)
            )
            bioclip_array[offset : offset + count] = (
                batch["bioclip_image"].numpy().astype(numpy_dtype, copy=False)
            )
            offset += count
        if observed != names:
            raise RuntimeError("Deterministic image cache loader changed filename order")
        dino_array.flush()
        bioclip_array.flush()
        manifest = {
            "version": CACHE_VERSION,
            "filenames": names,
            "dino_model_name": dino_model_name,
            "bioclip_checkpoint": bioclip_checkpoint,
            "dino_transform_hash": dino_transform_hash,
            "bioclip_transform_hash": bioclip_transform_hash,
            "dtype": dtype,
            "dino_shape": list(dino_array.shape),
            "bioclip_shape": list(bioclip_array.shape),
        }
        write_json(temporary / "manifest.json", manifest)
        del dino_array, bioclip_array
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_deterministic_image_cache(
        destination,
        expected_filenames=names,
        dino_model_name=dino_model_name,
        bioclip_checkpoint=bioclip_checkpoint,
        dino_transform_hash=dino_transform_hash,
        bioclip_transform_hash=bioclip_transform_hash,
    )


def load_image_cache_collection(
    root: str | Path,
    *,
    required_filenames: list[str],
    dino_model_name: str,
    bioclip_checkpoint: str,
    dino_transform_hash: str,
    bioclip_transform_hash: str,
) -> DeterministicImageCacheCollection:
    """Load only split caches present locally and require full requested coverage."""
    cache_root = Path(root)
    caches: list[DeterministicImageCache] = []
    for split in ("train", "test", "unseen"):
        path = cache_root / split
        if not path.exists():
            continue
        manifest = read_json(path / "manifest.json")
        caches.append(
            load_deterministic_image_cache(
                path,
                expected_filenames=list(manifest["filenames"]),
                dino_model_name=dino_model_name,
                bioclip_checkpoint=bioclip_checkpoint,
                dino_transform_hash=dino_transform_hash,
                bioclip_transform_hash=bioclip_transform_hash,
            )
        )
    collection = DeterministicImageCacheCollection(caches)
    collection.require(required_filenames)
    return collection
