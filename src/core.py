"""
core.py – Core physics and time-evolution functions.
=====================================================

Pure transverse coherence decay:
    L(t) = exp(-t / T2)

Bloch vector + Larmor precession:
    Mx(t) = M0 * cos(ω₀ t) * [exp(-t/T2)]
    My(t) = M0 * sin(ω₀ t) * [exp(-t/T2)]
    Mz(t) = Mz0  (constant — no T1 in P2)

    Transverse magnitude without T2:  √(Mx² + My²) = M0  (constant)
    Transverse magnitude with T2:     √(Mx² + My²) = M0·exp(-t/T2)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Tuple, Optional


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


# ===========================================================================
# PROTOTYPE 2 – Bloch vector & Larmor precession
# ===========================================================================

def bloch_precession(
    t,
    M0: float = 1.0,
    omega0: float = 2 * np.pi * 1e3,
    T2=None,
    Mz0: float = 0.0,
):
    """Compute Bloch vector components during Larmor precession.

    The spin starts fully in the transverse plane after an ideal pi/2 pulse:
        M(0) = [M0, 0, Mz0]

    Precession around z-axis at angular frequency omega0:
        Mx(t) = M0 * cos(omega0 * t)
        My(t) = M0 * sin(omega0 * t)
        Mz(t) = Mz0  (constant; T1 added in Prototype 3)

    With optional T2 dephasing:
        Mx(t) = M0 * cos(omega0 * t) * exp(-t / T2)
        My(t) = M0 * sin(omega0 * t) * exp(-t / T2)

    Parameters
    ----------
    t      : np.ndarray  time array (same units as 1/omega0 and T2)
    M0     : float       initial transverse magnitude          (default 1.0)
    omega0 : float       Larmor angular frequency rad/[t_unit] (default 2pi*1e3)
    T2     : float|None  transverse relaxation time; None = no decay
    Mz0    : float       initial longitudinal component        (default 0)

    Returns
    -------
    Mx, My, Mz : np.ndarray, shape (N,) each

    Invariants (for test suite)
    ---------------------------
    No T2 :  sqrt(Mx**2 + My**2)  == M0  everywhere (to machine precision)
    T2    :  sqrt(Mx**2 + My**2)  == M0 * exp(-t/T2)
    Always:  Mz == Mz0  (flat)
    t = 0 :  Mx = M0, My = 0, Mz = Mz0
    """
    t = np.asarray(t, dtype=float)
    if omega0 < 0:
        raise ValueError(f"omega0 must be non-negative, got {omega0}")
    if T2 is not None and T2 <= 0:
        raise ValueError(f"T2 must be positive when supplied, got {T2}")

    decay = simple_T2_decay(t, T2) if T2 is not None else np.ones_like(t)

    Mx = M0 * np.cos(omega0 * t) * decay
    My = M0 * np.sin(omega0 * t) * decay
    Mz = np.full_like(t, float(Mz0))

    return Mx, My, Mz


def plot_bloch_components(
    t,
    Mx,
    My,
    Mz,
    T2=None,
    omega0=None,
    time_unit: str = "µs",
    title: str = "Bloch Vector Components",
    save_path=None,
):
    """Four-panel figure: Mx(t), My(t), Mz(t), transverse magnitude |M_perp|(t).

    Panels
    ------
    Top-left  : Mx(t) with ±decay envelope
    Top-right : My(t) with ±decay envelope
    Bot-left  : Mz(t) (flat in P2, no T1)
    Bot-right : sqrt(Mx^2 + My^2) vs T2 exp decay overlay

    Parameters
    ----------
    t, Mx, My, Mz : arrays from bloch_precession()
    T2            : draws envelope + (T2, 1/e) marker when given
    omega0        : used for frequency annotation (optional)
    time_unit     : axis label suffix
    title         : figure suptitle
    save_path     : save PNG if given
    """
    M_perp = np.sqrt(Mx**2 + My**2)

    C_MX   = "#E63946"
    C_MY   = "#2C7BB6"
    C_MZ   = "#6A994E"
    C_PERP = "#9B2226"
    C_ENV  = "#F4A261"
    C_REF  = "#457B9D"

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    ax_mx, ax_my = axes[0, 0], axes[0, 1]
    ax_mz, ax_mp = axes[1, 0], axes[1, 1]

    def _envelope(ax, color):
        if T2 is not None:
            env = np.exp(-t / T2)
            ax.plot(t,  env, "--", color=color, lw=1.1, alpha=0.7, label="envelope")
            ax.plot(t, -env, "--", color=color, lw=1.1, alpha=0.7)
            ax.fill_between(t, -env, env, color=color, alpha=0.07)

    def _style(ax, ylabel):
        ax.axhline(0, color="black", lw=0.6, alpha=0.4)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(-1.15, 1.15)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_facecolor("#F9F9F9")

    # Mx
    ax_mx.plot(t, Mx, color=C_MX, lw=1.8, label=r"$M_x(t)$")
    _envelope(ax_mx, C_ENV)
    _style(ax_mx, r"$M_x$")
    ax_mx.set_title(r"$M_x = M_0\cos(\omega_0 t)\,e^{-t/T_2}$", fontsize=9)

    # My
    ax_my.plot(t, My, color=C_MY, lw=1.8, label=r"$M_y(t)$")
    _envelope(ax_my, C_ENV)
    _style(ax_my, r"$M_y$")
    ax_my.set_title(r"$M_y = M_0\sin(\omega_0 t)\,e^{-t/T_2}$", fontsize=9)

    # Mz
    ax_mz.plot(t, Mz, color=C_MZ, lw=2.0, label=r"$M_z(t)$")
    _style(ax_mz, r"$M_z$")
    ax_mz.set_title(r"$M_z(t) = M_{z0}$  (no $T_1$ yet)", fontsize=9)
    ax_mz.set_xlabel(f"Time  ({time_unit})", fontsize=11)

    # |M_perp|
    ax_mp.plot(t, M_perp, color=C_PERP, lw=2.0,
               label=r"$|M_\perp| = \sqrt{M_x^2+M_y^2}$")
    if T2 is not None:
        env = np.exp(-t / T2)
        ax_mp.plot(t, env, "--", color=C_REF, lw=1.4, alpha=0.8,
                   label=rf"$e^{{-t/T_2}}$  ($T_2$={T2} {time_unit})")
        idx = np.argmin(np.abs(t - T2))
        ax_mp.plot(t[idx], M_perp[idx], "o", color=C_REF, ms=7, zorder=6)
        ax_mp.annotate(
            r"$(T_2,\;1/e)$",
            xy=(t[idx], M_perp[idx]),
            xytext=(t[idx] + t[-1] * 0.06, M_perp[idx] + 0.08),
            fontsize=8, color=C_REF,
            arrowprops=dict(arrowstyle="->", color=C_REF, lw=0.9),
        )
    ax_mp.axhline(0, color="black", lw=0.6, alpha=0.4)
    ax_mp.set_ylim(-0.05, 1.15)
    ax_mp.set_ylabel(r"$|M_\perp|$", fontsize=11)
    ax_mp.set_title(r"Transverse magnitude  $|M_\perp|(t)$", fontsize=9)
    ax_mp.set_xlabel(f"Time  ({time_unit})", fontsize=11)
    ax_mp.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax_mp.grid(True, linestyle="--", alpha=0.35)
    ax_mp.set_facecolor("#F9F9F9")

    fig.patch.set_facecolor("white")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig