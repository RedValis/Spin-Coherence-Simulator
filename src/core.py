"""
core.py – Core physics and time-evolution functions.
=====================================================

Prototype 1 implements pure transverse coherence decay:

    L(t) = exp(-t / T2)

No precession, no T1 relaxation. I'm trying to have Future prototypes that will layer
in Bloch-equation evolution, T1, and pulse sequences. But I will see if that's possible
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Tuple


# ---------------------------------------------------------------------------
# Time axis
# ---------------------------------------------------------------------------

def time_axis(t_max: float, dt: float) -> np.ndarray:
    """Return a 1-D NumPy array of evenly-spaced times from 0 to t_max.

    Parameters
    ----------
    t_max : float
        End time (same units as dt, typically µs or ns).
    dt : float
        Time step.

    Returns
    -------
    np.ndarray
        Shape ``(N,)`` where N = floor(t_max / dt) + 1.

    Examples
    --------
    >>> t = time_axis(10.0, 1.0)
    >>> t
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
    """
    if t_max <= 0:
        raise ValueError(f"t_max must be positive, got {t_max}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    return np.arange(0.0, t_max + dt * 0.5, dt)   # inclusive of t_max


# ---------------------------------------------------------------------------
# Coherence decay
# ---------------------------------------------------------------------------

def simple_T2_decay(t: np.ndarray, T2: float) -> np.ndarray:
    """Compute pure T2 coherence decay: L(t) = exp(-t / T2).

    Parameters
    ----------
    t : array-like
        Time points (same units as T2).
    T2 : float
        Transverse relaxation time (coherence time).

    Returns
    -------
    np.ndarray
        Coherence values in [0, 1].  L(0) = 1, L(T2) ≈ 1/e ≈ 0.3679.

    Raises
    ------
    ValueError
        If T2 ≤ 0.
    """
    t = np.asarray(t, dtype=float)
    if T2 <= 0:
        raise ValueError(f"T2 must be positive, got {T2}")
    return np.exp(-t / T2)


# ---------------------------------------------------------------------------
# High-level simulation entry-point
# ---------------------------------------------------------------------------

def simulate_simple_coherence(
    T2: float,
    t_max: float,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate single-spin coherence decay under pure dephasing.

    Generates a time axis and evaluates L(t) = exp(-t / T2).  This is
    the minimal working model: the spin starts fully coherent in the
    transverse plane (L(0) = 1) and decays monotonically.

    Parameters
    ----------
    T2 : float
        Coherence / dephasing time (µs, ns, or any consistent unit).
    t_max : float
        Maximum simulation time.  Meaningful range: a few × T2.
    dt : float
        Time step.

    Returns
    -------
    t : np.ndarray
        Time array of shape ``(N,)``.
    L : np.ndarray
        Coherence array of shape ``(N,)`` with values in (0, 1].

    Quick checks
    ------------
    * ``L[0] == 1``             spin starts fully coherent
    * ``L[t == T2] ≈ 1/e``     half-life point
    * Monotonically decreasing  no revivals in pure T2 decay

    Examples
    --------
    >>> t, L = simulate_simple_coherence(T2=5.0, t_max=20.0, dt=0.1)
    >>> float(L[0])
    1.0
    >>> abs(L[t == 5.0][0] - 1/np.e) < 1e-10
    True
    """
    t = time_axis(t_max, dt)
    L = simple_T2_decay(t, T2)
    return t, L


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_coherence_decay(
    t: np.ndarray,
    L: np.ndarray,
    T2: float | None = None,
    title: str = "Simple T₂ Decay",
    time_unit: str = "µs",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot coherence L(t) vs time with physics annotations.

    Parameters
    ----------
    t : np.ndarray
        Time axis produced by :func:`time_axis`.
    L : np.ndarray
        Coherence array produced by :func:`simple_T2_decay`.
    T2 : float, optional
        If supplied, draw the characteristic T2 marker at (T2, 1/e).
    title : str
        Plot title.
    time_unit : str
        Label for the horizontal axis (default ``"µs"``).
    save_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # --- main decay curve ---------------------------------------------------
    ax.plot(t, L, color="#2C7BB6", linewidth=2.2, label=r"$L(t) = e^{-t/T_2}$")

    # --- 1/e reference line -------------------------------------------------
    ax.axhline(1 / np.e, color="#D7191C", linewidth=1.2,
               linestyle="--", alpha=0.85, label=r"$L = 1/e \approx 0.368$")

    # --- T2 marker ----------------------------------------------------------
    if T2 is not None:
        ax.axvline(T2, color="#1A9641", linewidth=1.2,
                   linestyle=":", alpha=0.85, label=rf"$T_2 = {T2}$ {time_unit}")
        ax.plot(T2, 1 / np.e, "o", color="#1A9641",
                markersize=8, zorder=5)
        ax.annotate(
            rf"$(T_2,\; 1/e)$",
            xy=(T2, 1 / np.e),
            xytext=(T2 + (t[-1] - t[0]) * 0.04, 1 / np.e + 0.04),
            fontsize=9,
            color="#1A9641",
            arrowprops=dict(arrowstyle="->", color="#1A9641", lw=1.0),
        )

    # --- L(0) = 1 marker ----------------------------------------------------
    ax.plot(0, 1.0, "^", color="#2C7BB6", markersize=8, zorder=5,
            label=r"$L(0) = 1$")

    # --- formatting ---------------------------------------------------------
    ax.set_xlim(left=0)
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlabel(f"Time  ({time_unit})", fontsize=12)
    ax.set_ylabel(r"Coherence  $L(t)$", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_facecolor("#F9F9F9")
    fig.patch.set_facecolor("white")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig