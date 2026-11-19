"""Pre-baked Matplotlib-compatible colormap lookup tables without Matplotlib at runtime.

The lookup table archive ``pySAview/resources/colormaps.npz`` stores 8-bit RGB
tables sampled directly from Matplotlib's built-in colormaps (base maps plus
exact reversed ``_r`` overrides where midpoint rounding differs from simple
reversal). This preserves all 182 Matplotlib colormap names and hex values
while keeping Matplotlib out of packaged runtime builds.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import math

import numpy as np

from pySAview.core.paths import resources_dir


class ColormapLUT:
    """Discrete RGB lookup table matching Matplotlib's ``Colormap`` indexing."""

    __slots__ = ("name", "_rgb", "N")

    def __init__(self, name: str, rgb: np.ndarray) -> None:
        self.name = name
        self._rgb = rgb
        self.N = int(rgb.shape[0])

    @property
    def rgb_uint8(self) -> np.ndarray:
        """Return the ``(N, 3)`` uint8 RGB lookup table."""
        return self._rgb

    def _index(self, position: float) -> int:
        val = float(position)
        if math.isnan(val) or val <= 0.0:
            return 0
        if val >= 1.0:
            return self.N - 1
        idx = int(val * self.N)
        return self.N - 1 if idx >= self.N else idx

    def __call__(self, position: float) -> tuple[float, float, float, float]:
        r, g, b = self._rgb[self._index(position)]
        return (int(r) / 255.0, int(g) / 255.0, int(b) / 255.0, 1.0)

    def to_hex(self, position: float) -> str:
        r, g, b = self._rgb[self._index(position)]
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def to_hex(color: tuple[float, ...] | list[float] | np.ndarray) -> str:
    """Convert an RGB or RGBA float sequence in ``[0, 1]`` to ``#rrggbb``."""
    r, g, b = color[0], color[1], color[2]
    ri = max(0, min(255, int(round(float(r) * 255.0))))
    gi = max(0, min(255, int(round(float(g) * 255.0))))
    bi = max(0, min(255, int(round(float(b) * 255.0))))
    return f"#{ri:02x}{gi:02x}{bi:02x}"


class _ColormapRegistry(Mapping[str, ColormapLUT]):
    """Lazy mapping of Matplotlib colormap names to ``ColormapLUT`` instances."""

    def __init__(self) -> None:
        self._loaded = False
        self._all_names: tuple[str, ...] = ()
        self._slices: dict[str, tuple[int, int]] = {}
        self._rgb: np.ndarray | None = None
        self._cache: dict[str, ColormapLUT] = {}

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        archive_path = resources_dir() / "colormaps.npz"
        with np.load(archive_path, allow_pickle=False) as data:
            self._all_names = tuple(str(n) for n in data["all_names"])
            stored_names = [str(n) for n in data["stored_names"]]
            offsets = data["offsets"]
            lengths = data["lengths"]
            self._rgb = np.array(data["rgb"], dtype=np.uint8, copy=True)
            self._slices = {
                name: (int(off), int(off) + int(length))
                for name, off, length in zip(stored_names, offsets, lengths)
            }
        self._loaded = True

    def __getitem__(self, name: str) -> ColormapLUT:
        self._ensure_loaded()
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        assert self._rgb is not None
        if name in self._slices:
            start, stop = self._slices[name]
            lut = self._rgb[start:stop]
        elif name.endswith("_r") and name[:-2] in self._slices:
            start, stop = self._slices[name[:-2]]
            lut = np.ascontiguousarray(self._rgb[start:stop][::-1])
        else:
            raise KeyError(f"Unknown colormap: {name!r}")
        cmap = ColormapLUT(name, lut)
        self._cache[name] = cmap
        return cmap

    def __iter__(self) -> Iterator[str]:
        self._ensure_loaded()
        return iter(self._all_names)

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._all_names)

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        self._ensure_loaded()
        return name in self._slices or (name.endswith("_r") and name[:-2] in self._slices)


colormaps = _ColormapRegistry()
