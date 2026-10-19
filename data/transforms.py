from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
from scipy.ndimage import rotate


class Transform:
    """Base transform for paired arrays with leading batch dimension.

    Image shape: ``(N,H,W,D,C)``. Label shape: ``(N,H,W,D)``.
    ``k=0`` denotes the image and ``k=1`` the segmentation label.
    """

    def sample(self, *shape: int) -> Tuple[int, ...]:
        return tuple(shape)

    def apply(self, array: np.ndarray, k: int = 0) -> np.ndarray:
        return array

    def __call__(self, arrays, dim: int = 3, reuse: bool = False):
        sequence = isinstance(arrays, Sequence) and not isinstance(arrays, np.ndarray)
        reference = arrays[0] if sequence else arrays
        if not reuse:
            self.sample(*reference.shape[1 : dim + 1])
        if sequence:
            return [self.apply(value, k=index) for index, value in enumerate(arrays)]
        return self.apply(arrays, k=0)


class Identity(Transform):
    pass


class Compose(Transform):
    def __init__(self, transforms: Iterable[Transform]):
        self.transforms = list(transforms)

    def sample(self, *shape: int) -> Tuple[int, ...]:
        current = tuple(shape)
        for transform in self.transforms:
            current = tuple(transform.sample(*current))
        return current

    def apply(self, array: np.ndarray, k: int = 0) -> np.ndarray:
        output = array
        for transform in self.transforms:
            output = transform.apply(output, k=k)
        return output


class RandomCrop3D(Transform):
    def __init__(self, size: Tuple[int, int, int]):
        if len(size) != 3:
            raise ValueError("RandomCrop3D requires a 3-D size.")
        self.size = tuple(int(v) for v in size)
        if min(self.size) <= 0:
            raise ValueError("Crop dimensions must be positive.")
        self.slices = None

    def sample(self, *shape: int) -> Tuple[int, ...]:
        if len(shape) != 3:
            raise ValueError(f"Expected a 3-D spatial shape, got {shape}")
        starts = []
        for length, crop in zip(shape, self.size):
            if length < crop:
                raise ValueError(
                    f"Volume shape {shape} is smaller than crop {self.size}. "
                    "Run preprocessing with minimum support >= crop size."
                )
            starts.append(random.randint(0, length - crop))
        self.slices = (slice(None),) + tuple(
            slice(start, start + crop) for start, crop in zip(starts, self.size)
        )
        return self.size

    def apply(self, array: np.ndarray, k: int = 0) -> np.ndarray:
        if self.slices is None:
            raise RuntimeError("RandomCrop3D.sample must run before apply.")
        return array[self.slices]


class RandomRotation3D(Transform):
    """Paired 3-D rotation with interpolation appropriate to each tensor.

    MRI channels use linear interpolation and zero padding. Segmentation labels
    use nearest-neighbor interpolation and background padding. This prevents
    invalid negative labels and avoids turning padded voxels into ET during
    one-hot conversion.
    """

    def __init__(self, max_degrees: int = 10, probability: float = 1.0):
        if max_degrees < 0:
            raise ValueError("max_degrees must be non-negative.")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must lie in [0,1].")
        self.max_degrees = int(max_degrees)
        self.probability = float(probability)
        self.axes_options = ((1, 0), (2, 1), (2, 0))
        self.enabled = False
        self.angle = 0.0
        self.axes = self.axes_options[0]

    def sample(self, *shape: int) -> Tuple[int, ...]:
        self.enabled = random.random() < self.probability and self.max_degrees > 0
        if self.enabled:
            self.axes = self.axes_options[np.random.randint(len(self.axes_options))]
            self.angle = float(np.random.uniform(-self.max_degrees, self.max_degrees))
        else:
            self.angle = 0.0
        return tuple(shape)

    def apply(self, array: np.ndarray, k: int = 0) -> np.ndarray:
        if not self.enabled or abs(self.angle) < 1e-8:
            return array

        output = np.empty_like(array)
        if k == 0:
            if array.ndim != 5:
                raise ValueError(f"Expected image shape (N,H,W,D,C), got {array.shape}")
            for batch in range(array.shape[0]):
                channels = []
                for channel in range(array.shape[-1]):
                    channels.append(
                        rotate(
                            array[batch, ..., channel],
                            angle=self.angle,
                            axes=self.axes,
                            reshape=False,
                            order=1,
                            mode="constant",
                            cval=0.0,
                            prefilter=False,
                        )
                    )
                output[batch] = np.stack(channels, axis=-1)
            return output

        if k == 1:
            if array.ndim != 4:
                raise ValueError(f"Expected label shape (N,H,W,D), got {array.shape}")
            for batch in range(array.shape[0]):
                output[batch] = rotate(
                    array[batch],
                    angle=self.angle,
                    axes=self.axes,
                    reshape=False,
                    order=0,
                    mode="constant",
                    cval=0,
                    prefilter=False,
                )
            return output

        return array


class RandomIntensityChange(Transform):
    def __init__(self, shift: float = 0.1, scale: float = 0.1):
        if shift < 0 or scale < 0:
            raise ValueError("shift and scale amplitudes must be non-negative.")
        self.shift = float(shift)
        self.scale = float(scale)
        self.scale_values = None
        self.shift_values = None

    def sample(self, *shape: int) -> Tuple[int, ...]:
        self.scale_values = np.random.uniform(1.0 - self.scale, 1.0 + self.scale, size=(1, 1, 1, 1, 4))
        self.shift_values = np.random.uniform(-self.shift, self.shift, size=(1, 1, 1, 1, 4))
        return tuple(shape)

    def apply(self, array: np.ndarray, k: int = 0) -> np.ndarray:
        if k != 0:
            return array
        if array.ndim != 5 or array.shape[-1] != 4:
            raise ValueError(f"Expected image shape (N,H,W,D,4), got {array.shape}")
        foreground = np.abs(array) > 1e-8
        changed = array * self.scale_values + self.shift_values
        return np.where(foreground, changed, 0.0).astype(array.dtype, copy=False)


class RandomFlip3D(Transform):
    def __init__(self, probability: float = 0.5):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must lie in [0,1].")
        self.probability = float(probability)
        self.flags = (False, False, False)

    def sample(self, *shape: int) -> Tuple[int, ...]:
        self.flags = tuple(random.random() < self.probability for _ in range(3))
        return tuple(shape)

    def apply(self, array: np.ndarray, k: int = 0) -> np.ndarray:
        output = array
        # Leading dimension is batch; spatial axes are 1,2,3 for both inputs.
        for axis, enabled in zip((1, 2, 3), self.flags):
            if enabled:
                output = np.flip(output, axis=axis)
        return output


class NumpyType(Transform):
    def __init__(self, dtypes):
        self.dtypes = tuple(dtypes)

    def apply(self, array: np.ndarray, k: int = 0) -> np.ndarray:
        if k >= len(self.dtypes):
            return array
        return array.astype(self.dtypes[k], copy=False)


# Backward-compatible aliases used by the older project.
RandCrop3D = RandomCrop3D
RandomRotion = RandomRotation3D
RandomFlip = RandomFlip3D
