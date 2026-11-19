"""Polarized neutron spin-state selection, pairing, pooling, and asymmetry.

Follows the SuperADAM flipper convention:
- ``Flipper1 == 0``: flipper ON -> spin-down ``d``
- ``Flipper1 == 1``: flipper OFF -> spin-up ``u``
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pySAview.core.reduction.normalization import (
    NORMALIZATION_UNITS,
    NormalizedMap,
    normalize_map,
)


def select_frames(n_frames: int, flipper: np.ndarray | None, spin: str) -> np.ndarray:
    """Frame indices for 'all', 'up' (Flipper1 == 1, OFF) or 'down' (Flipper1 == 0, ON)."""
    every = np.arange(n_frames)
    if spin == "all" or flipper is None:
        return every
    state = np.asarray(flipper, dtype=float)[:n_frames]
    wanted = 1.0 if spin == "up" else 0.0
    return np.flatnonzero(state == wanted)


def split_indices_by_flipper(
    x_data: np.ndarray,
    flipper_data: np.ndarray,
    spin_mode: str = "split",
) -> list[tuple[str | None, str, np.ndarray]]:
    """Split scan point indices by flipper state without UI or color dependencies.

    Parameters
    ----------
    x_data:
        1D array of independent-axis coordinates used to sort split spin channels.
    flipper_data:
        1D array of flipper state values (``0`` = ``d (ON)``, ``1`` = ``u (OFF)``).
    spin_mode:
        - ``'all'``: single group ``(None, 'All states', unsorted_indices)``
        - ``'split'``: up to two groups (``'d'`` then ``'u'``), each sorted by ``x_data``;
          falls back to ``(None, 'Scan', unsorted_indices)`` if neither state is present
        - ``'flipper0'``: only ``'d'`` (Flipper1 == 0), or ``[]`` if absent
        - ``'flipper1'``: only ``'u'`` (Flipper1 == 1), or ``[]`` if absent

    Returns
    -------
    list[tuple[str | None, str, np.ndarray]]
        Each entry is ``(spin_key, label, indices)``.
    """
    x = np.asarray(x_data)
    indices = np.arange(len(x))
    if spin_mode == "all":
        return [(None, "All states", indices)]

    f = np.asarray(flipper_data)
    mask0 = f == 0
    mask1 = f == 1

    groups: list[tuple[str | None, str, np.ndarray]] = []
    if spin_mode in ("split", "flipper0") and np.any(mask0):
        idx0 = indices[mask0]
        sorted_idx0 = idx0[np.argsort(x[mask0])]
        groups.append(("d", "d (ON)", sorted_idx0))

    if spin_mode in ("split", "flipper1") and np.any(mask1):
        idx1 = indices[mask1]
        sorted_idx1 = idx1[np.argsort(x[mask1])]
        groups.append(("u", "u (OFF)", sorted_idx1))

    if not groups:
        if spin_mode in ("flipper0", "flipper1"):
            return []
        return [(None, "Scan", indices)]

    return groups


def pair_spin_frames(th, tth, flipper, written=None, tol_deg: float = 2e-3):
    """Pair each spin-up frame (Flipper1 = 1, OFF) with an unused spin-down frame (0, ON) at the same θ and 2θ.

    The nearest-in-sequence down frame within ``tol_deg`` in both angles is
    used; unwritten frames are never paired. Returns (up, down) index arrays.
    """
    th = np.asarray(th, dtype=float)
    tth = np.asarray(tth, dtype=float)
    flipper = np.asarray(flipper, dtype=float)
    n = min(th.size, tth.size, flipper.size)
    usable = np.isfinite(th[:n]) & np.isfinite(tth[:n])
    if written is not None:
        usable &= np.asarray(written, dtype=bool)[:n]
    ups = np.flatnonzero(usable & (flipper[:n] == 1))
    downs = np.flatnonzero(usable & (flipper[:n] == 0))
    taken = np.zeros(downs.size, dtype=bool)
    pairs = []
    for u in ups:
        match = (~taken & (np.abs(th[downs] - th[u]) <= tol_deg) & (np.abs(tth[downs] - tth[u]) <= tol_deg))
        if not np.any(match):
            continue
        candidates = np.flatnonzero(match)
        k = candidates[np.argmin(np.abs(downs[candidates] - u))]
        taken[k] = True
        pairs.append((u, downs[k]))
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    return pairs[:, 0], pairs[:, 1]


@dataclass
class SummedSpinMap(NormalizedMap):
    """Pooled u + d spin channels: ``frame_indices`` are the up frames, ``partner_indices`` the down frames."""

    partner_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))


def sum_spin_frames(counts, up, down, mode: str = "raw", *, exposure=None, monitor=None,
                    monitor_variance=None) -> SummedSpinMap:
    """Per-pixel pooled intensity of paired spin-up and spin-down frames.

    raw:     N_u + N_d,                   var = N_u + N_d
    time:    (N_u + N_d) / (t_u + t_d),   var = (N_u + N_d) / (t_u + t_d)**2
    monitor: (N_u + N_d) / (M_u + M_d),   var = N_tot / M_tot**2 + N_tot**2 * v_M / M_tot**4
             where N_tot = N_u + N_d, M_tot = M_u + M_d, and v_M = M_tot (or v_Mu + v_Md).

    Pairs where either frame has a missing, zero, negative or nonfinite
    denominator (or invalid monitor variance) become NaN and are reported in
    ``warnings``; nothing is clipped or substituted.
    """
    if mode not in NORMALIZATION_UNITS:
        raise ValueError(f"Unknown normalization mode: {mode}")
    up = np.asarray(up, dtype=int)
    down = np.asarray(down, dtype=int)
    if up.shape != down.shape:
        raise ValueError("Paired spin index arrays must have matching length")
    raw = np.asarray(counts, dtype=float)
    n_tot = raw[up] + raw[down]
    warnings: list[str] = []
    variance = n_tot.copy()

    if mode == "raw":
        values = n_tot.copy()
    else:
        source = exposure if mode == "time" else monitor
        name = "exposure time" if mode == "time" else "Monitor1"
        if source is None:
            raise ValueError(f"{name} is required for {mode} normalization")
        source = np.asarray(source, dtype=float)
        max_idx = max(int(up.max()), int(down.max())) + 1 if up.size else 0
        if source.shape[0] < max_idx:
            raise ValueError(f"{name} has fewer points than the detector has frames")
        su, sd = source[up], source[down]
        valid = np.isfinite(su) & (su > 0) & np.isfinite(sd) & (sd > 0)
        safe = np.where(valid, su + sd, np.nan)[:, None]
        if mode == "time":
            values = n_tot / safe
            variance = n_tot / safe**2
        else:
            v_m = safe
            if monitor_variance is not None:
                mv = np.asarray(monitor_variance, dtype=float)
                if mv.shape[0] < max_idx:
                    raise ValueError("Monitor1 variance has fewer points than the detector has frames")
                v_mu, v_md = mv[up], mv[down]
                bad_var = ~(np.isfinite(v_mu) & (v_mu >= 0) & np.isfinite(v_md) & (v_md >= 0))
                valid = valid & ~bad_var
                safe = np.where(valid, su + sd, np.nan)[:, None]
                v_m = np.where(valid, v_mu + v_md, np.nan)[:, None]
            values = np.where(valid[:, None], n_tot / safe, np.nan)
            variance = n_tot / safe**2 + n_tot**2 * v_m / safe**4
        bad_rows = int(np.count_nonzero(~valid))
        if bad_rows:
            warnings.append(f"{bad_rows} point(s) excluded: invalid {name} or its variance.")

    return SummedSpinMap(
        frame_indices=up,
        values=values,
        standard_uncertainty=np.sqrt(variance),
        mode=mode,
        units=NORMALIZATION_UNITS[mode],
        warnings=warnings,
        partner_indices=down,
    )


@dataclass
class AsymmetryMap(NormalizedMap):
    """Spin asymmetry rows: ``frame_indices`` are the up frames, ``partner_indices`` the down frames."""

    partner_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    low_count_mask: np.ndarray | None = None  # True where raw N_u + N_d is below the display threshold
    min_counts: float = 0.0


def spin_asymmetry(counts, up, down, mode: str = "raw", *, exposure=None, monitor=None,
                   monitor_variance=None, min_counts: float = 0.0) -> AsymmetryMap:
    """Per-pixel spin asymmetry SA = (I_u − I_d) / (I_u + I_d) of paired frames.

    I_u and I_d are the selected normalization of each frame (raw, per second or
    per monitor, as in ``normalize_map``), so unequal exposures cancel when a
    normalization is chosen. First-order 1σ uncertainty for independent u/d:

        var(SA) = (2 I_d / S²)² var(I_u) + (2 I_u / S²)² var(I_d),  S = I_u + I_d

    Pixels with S ≤ 0 or invalid normalization are NaN. ``low_count_mask``
    flags pixels with fewer than ``min_counts`` raw counts in u + d; it is for
    display only and does not change the values.
    """
    up = np.asarray(up, dtype=int)
    down = np.asarray(down, dtype=int)
    kw = dict(exposure=exposure, monitor=monitor, monitor_variance=monitor_variance)
    iu = normalize_map(counts, up, mode, **kw)
    idn = normalize_map(counts, down, mode, **kw)
    s = iu.values + idn.values
    valid = np.isfinite(s) & (s > 0)
    safe = np.where(valid, s, np.nan)
    sa = (iu.values - idn.values) / safe
    var = ((2 * idn.values / safe**2) ** 2 * iu.standard_uncertainty**2
           + (2 * iu.values / safe**2) ** 2 * idn.standard_uncertainty**2)
    raw = np.asarray(counts, dtype=float)
    low = (raw[up] + raw[down]) < min_counts
    warnings = list(dict.fromkeys(iu.warnings + idn.warnings))
    return AsymmetryMap(frame_indices=up, values=sa, standard_uncertainty=np.sqrt(var), mode=mode,
                        units="spin asymmetry (u−d)/(u+d)", warnings=warnings,
                        partner_indices=down, low_count_mask=low, min_counts=float(min_counts))


__all__ = [
    "AsymmetryMap",
    "SummedSpinMap",
    "pair_spin_frames",
    "select_frames",
    "spin_asymmetry",
    "split_indices_by_flipper",
    "sum_spin_frames",
]
