"""
sequences.py - Pulse sequences for spin coherence simulation.
=============================================================

Prototype 4 implements:
  - apply_pulse        : instantaneous rotation via SO(3) rotation matrices
  - free_evolve        : Bloch ODE integration over a time segment
  - hahn_echo_sequence : π/2 → τ → π → τ → echo
  - cpmg_sequence      : π/2 → [τ → π → τ]xN (multi-echo train)
  - sweep_echo_amplitude: echo amplitude vs 2τ (decay curve)
  - measure_echo_amplitude: peak |M_perp| near echo time
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, List
from .core import simulate_bloch, time_axis


# ===========================================================================
# Rotation matrices (SO(3))
# ===========================================================================

def _Rx(theta: float) -> np.ndarray:
    """3x3 rotation matrix around x-axis by angle theta (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0,  0],
                     [0, c, -s],
                     [0, s,  c]])

def _Ry(theta: float) -> np.ndarray:
    """3x3 rotation matrix around y-axis by angle theta (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def _Rz(theta: float) -> np.ndarray:
    """3x3 rotation matrix around z-axis by angle theta (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

_ROT = {'x': _Rx, 'y': _Ry, 'z': _Rz,
        '-x': lambda t: _Rx(-t),
        '-y': lambda t: _Ry(-t),
        '-z': lambda t: _Rz(-t)}


# ===========================================================================
# Core building blocks
# ===========================================================================

def apply_pulse(
    M: np.ndarray,
    axis: str = 'x',
    angle: float = np.pi / 2,
) -> np.ndarray:
    """Rotate Bloch vector M by *angle* around *axis* (instantaneous pulse).

    Implements hard-pulse approximation: the pulse is infinitely short
    compared to T1, T2, and the Larmor period. The rotation is exact via
    SO(3) rotation matrices, so there is no numerical error in the pulse.

    Parameters
    ----------
    M     : (3,) array  current Bloch vector [Mx, My, Mz]
    axis  : str         rotation axis — 'x', 'y', 'z', '-x', '-y', '-z'
    angle : float       rotation angle in radians
                        π/2 = 90° tip pulse
                        π   = 180° inversion / refocusing pulse

    Returns
    -------
    M_rotated : (3,) np.ndarray

    Examples
    --------
    >>> M = np.array([0., 0., 1.])         # spin along +z (equilibrium)
    >>> apply_pulse(M, axis='y', angle=np.pi/2)   # tip to +x
    array([1., 0., 0.])
    >>> apply_pulse(M, axis='x', angle=np.pi)     # invert to -z
    array([0., 0., -1.])

    Invariants
    ----------
    * |M_rotated| == |M|  (rotation preserves vector length)
    * Two π pulses around the same axis → identity
    """
    M = np.asarray(M, dtype=float)
    if axis not in _ROT:
        raise ValueError(f"axis must be one of {list(_ROT)}, got '{axis}'")
    R = _ROT[axis](angle)
    return R @ M


def free_evolve(
    M_init: np.ndarray,
    t_free: float,
    dt: float,
    gamma: float,
    B: np.ndarray,
    T1: float,
    T2: float,
    M0: float,
    method: str = "RK45",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evolve M under free precession + T1/T2 relaxation for duration t_free.

    Thin wrapper around simulate_bloch that emphasises its role as a
    *segment* in a larger pulse sequence.  The returned time array
    starts at 0 (relative to the segment start); callers are responsible
    for stitching absolute times.

    Parameters
    ----------
    M_init  : (3,) array  initial Bloch vector for this segment
    t_free  : float       duration of free evolution
    dt      : float       time step
    gamma   : float       gyromagnetic ratio
    B       : (3,) array  magnetic field [Bx, By, Bz]
    T1, T2  : float       relaxation times
    M0      : float       equilibrium magnetisation
    method  : str         ODE solver (default 'RK45')

    Returns
    -------
    t_seg, Mx_seg, My_seg, Mz_seg : np.ndarray, shape (N,) each
        t_seg[0] = 0, t_seg[-1] ≈ t_free
    """
    t_seg, Mx, My, Mz = simulate_bloch(
        M_init=M_init,
        gamma=gamma, B=B,
        T1=T1, T2=T2, M0=M0,
        t_max=t_free, dt=dt,
        method=method,
    )
    return t_seg, Mx, My, Mz


# ===========================================================================
# Hahn echo
# ===========================================================================

def hahn_echo_sequence(
    gamma: float,
    B: np.ndarray,
    T1: float,
    T2: float,
    M0: float,
    tau: float,
    dt: float,
    tip_axis: str = 'y',
    refocus_axis: str = 'x',
    M_eq: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a Hahn spin-echo sequence.

    Sequence
    --------
    1.  π/2 pulse (tip_axis)     : M_eq → transverse plane
    2.  Free evolution τ          : spins dephase
    3.  π pulse (refocus_axis)   : dephased spins refocused
    4.  Free evolution τ          : spins rephase → echo at t = 2τ

    The echo at t = 2τ refocuses *inhomogeneous* dephasing (static field
    offsets) but NOT the irreversible T2 decay.  The echo amplitude is:

        |M_echo| = M0 · exp(-2τ / T2)

    Parameters
    ----------
    gamma        : float        gyromagnetic ratio
    B            : (3,) array   field vector (typically [0, 0, B0])
    T1, T2       : float        relaxation times (T2 ≤ T1)
    M0           : float        equilibrium magnetisation
    tau          : float        half-echo time (echo appears at 2τ)
    dt           : float        time step
    tip_axis     : str          axis for π/2 pulse (default 'y' → tips to +x)
    refocus_axis : str          axis for π refocusing pulse (default 'x')
    M_eq         : (3,) array   equilibrium state before π/2 (default [0,0,M0])

    Returns
    -------
    t_full, Mx, My, Mz : np.ndarray, shape (N,) each
        Concatenated time axis (0 → 2τ) and Bloch components.
        The echo appears near t = 2τ.

    Verification
    ------------
    * Echo amplitude ≈ M0 · exp(-2τ/T2)
    * No echo without the π pulse (just monotonic decay)
    * T2 >> tau → echo amplitude ≈ M0
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    if dt >= tau:
        raise ValueError(f"dt ({dt}) must be smaller than tau ({tau})")

    M_eq = np.array([0.0, 0.0, M0]) if M_eq is None else np.asarray(M_eq, dtype=float)

    # ── Segment 1: π/2 pulse ────────────────────────────────────────────────
    M_after_pi2 = apply_pulse(M_eq, axis=tip_axis, angle=np.pi / 2)

    # ── Segment 2: free evolution τ ──────────────────────────────────────────
    t1, Mx1, My1, Mz1 = free_evolve(M_after_pi2, tau, dt, gamma, B, T1, T2, M0)
    M_end_seg1 = np.array([Mx1[-1], My1[-1], Mz1[-1]])

    # ── Segment 3: π refocusing pulse ────────────────────────────────────────
    M_after_pi = apply_pulse(M_end_seg1, axis=refocus_axis, angle=np.pi)

    # ── Segment 4: free evolution τ ──────────────────────────────────────────
    t2, Mx2, My2, Mz2 = free_evolve(M_after_pi, tau, dt, gamma, B, T1, T2, M0)

    # ── Stitch segments (avoid duplicate boundary point) ─────────────────────
    t_full = np.concatenate([t1, t1[-1] + t2[1:]])
    Mx     = np.concatenate([Mx1, Mx2[1:]])
    My     = np.concatenate([My1, My2[1:]])
    Mz     = np.concatenate([Mz1, Mz2[1:]])

    return t_full, Mx, My, Mz


# ===========================================================================
# CPMG multi-echo train
# ===========================================================================

def cpmg_sequence(
    gamma: float,
    B: np.ndarray,
    T1: float,
    T2: float,
    M0: float,
    tau: float,
    n_echoes: int,
    dt: float,
    tip_axis: str = 'y',
    refocus_axis: str = 'x',
    M_eq: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a CPMG (Carr-Purcell-Meiboom-Gill) multi-echo sequence.

    Sequence
    --------
    π/2 → [τ → π → τ → (echo)]xn_echoes

    Each π pulse refocuses the spins.  The echo amplitudes form a staircase
    decaying as exp(-t_echo / T2), where t_echo = 2·k·τ for the k-th echo.
    CPMG is more robust than repeated Hahn echoes because the Meiboom-Gill
    phase shift (tip on y, refocus on x) corrects pulse imperfections.

    Parameters
    ----------
    gamma, B, T1, T2, M0 : as in hahn_echo_sequence
    tau      : float   half-spacing between π pulses
    n_echoes : int     number of refocusing pulses (and echoes)
    dt       : float   time step
    tip_axis, refocus_axis : str  pulse axes

    Returns
    -------
    t, Mx, My, Mz : full time axis and components
    echo_times    : np.ndarray of shape (n_echoes,), echo centres at 2k·τ
    """
    if n_echoes < 1:
        raise ValueError(f"n_echoes must be ≥ 1, got {n_echoes}")
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    if dt >= tau:
        raise ValueError(f"dt ({dt}) must be smaller than tau ({tau})")

    M_eq = np.array([0.0, 0.0, M0]) if M_eq is None else np.asarray(M_eq, dtype=float)

    # π/2 tip pulse
    M_cur = apply_pulse(M_eq, axis=tip_axis, angle=np.pi / 2)

    t_all  = np.array([0.0])
    Mx_all = np.array([M_cur[0]])
    My_all = np.array([M_cur[1]])
    Mz_all = np.array([M_cur[2]])
    t_offset   = 0.0
    echo_times = []

    for k in range(n_echoes):
        # free evolve τ
        t_s, Mx_s, My_s, Mz_s = free_evolve(M_cur, tau, dt, gamma, B, T1, T2, M0)
        M_cur = np.array([Mx_s[-1], My_s[-1], Mz_s[-1]])

        t_all  = np.concatenate([t_all,  t_offset + t_s[1:]])
        Mx_all = np.concatenate([Mx_all, Mx_s[1:]])
        My_all = np.concatenate([My_all, My_s[1:]])
        Mz_all = np.concatenate([Mz_all, Mz_s[1:]])
        t_offset += t_s[-1]

        # π refocusing pulse
        M_cur = apply_pulse(M_cur, axis=refocus_axis, angle=np.pi)

        # free evolve τ (echo forms at end of this segment)
        t_s2, Mx_s2, My_s2, Mz_s2 = free_evolve(M_cur, tau, dt, gamma, B, T1, T2, M0)
        M_cur = np.array([Mx_s2[-1], My_s2[-1], Mz_s2[-1]])

        t_all  = np.concatenate([t_all,  t_offset + t_s2[1:]])
        Mx_all = np.concatenate([Mx_all, Mx_s2[1:]])
        My_all = np.concatenate([My_all, My_s2[1:]])
        Mz_all = np.concatenate([Mz_all, Mz_s2[1:]])
        t_offset += t_s2[-1]

        echo_times.append(t_offset)

    return t_all, Mx_all, My_all, Mz_all, np.array(echo_times)


# ===========================================================================
# Echo amplitude measurement
# ===========================================================================

def measure_echo_amplitude(
    Mx: np.ndarray,
    My: np.ndarray,
    t: np.ndarray,
    echo_time: float,
    window: float = 0.1,
) -> float:
    """Return peak transverse magnitude |M_perp| near the echo time.

    Searches within ±window·(echo_time) of the expected echo centre to
    find the maximum |M_perp|.  A narrow window prevents picking up
    spurious earlier peaks.

    Parameters
    ----------
    Mx, My     : component arrays from hahn_echo_sequence
    t          : time axis
    echo_time  : expected echo centre (e.g. 2·tau for Hahn echo)
    window     : fractional search window around echo_time (default 0.1 = ±10%)

    Returns
    -------
    float  peak |M_perp| near the echo
    """
    M_perp   = np.sqrt(Mx**2 + My**2)
    half_win = window * echo_time
    mask     = np.abs(t - echo_time) <= half_win
    if not np.any(mask):
        # fall back to global max if window misses
        return float(np.max(M_perp))
    return float(np.max(M_perp[mask]))


def sweep_echo_amplitude(
    gamma: float,
    B: np.ndarray,
    T1: float,
    T2: float,
    M0: float,
    tau_values: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sweep τ and record Hahn echo amplitude at t = 2τ for each value.

    This generates the canonical T2 decay curve from echo experiments:

        A_echo(2τ) = M0 · exp(-2τ / T2)

    Parameters
    ----------
    gamma, B, T1, T2, M0 : physics parameters
    tau_values : 1-D array of τ values to sweep
    dt         : time step for each Hahn echo simulation

    Returns
    -------
    two_tau   : np.ndarray  echo times = 2·tau_values
    amplitudes: np.ndarray  measured echo amplitudes
    """
    amplitudes = []
    for tau in tau_values:
        t, Mx, My, Mz = hahn_echo_sequence(
            gamma=gamma, B=B, T1=T1, T2=T2, M0=M0, tau=tau, dt=dt)
        amp = measure_echo_amplitude(Mx, My, t, echo_time=2*tau)
        amplitudes.append(amp)
    return 2 * tau_values, np.array(amplitudes)