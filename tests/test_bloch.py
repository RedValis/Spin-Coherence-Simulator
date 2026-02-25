
"""
tests/test_bloch.py – Unit tests for Prototype 2 (Bloch vector & precession).
==============================================================================

Physics invariants under test:

  1. Without T2:  |M_perp| = sqrt(Mx^2 + My^2) == M0  everywhere
  2. With T2:     |M_perp| = M0 * exp(-t/T2)
  3. At t = 0:    Mx = M0,  My = 0,  Mz = Mz0
  4. Mz           is constant throughout (no T1)
  5. Input validation: omega0 < 0 raises, T2 = 0 raises

Run with:  pytest tests/test_bloch.py -v
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core import time_axis, bloch_precession

PASS = "  PASS"
FAIL = "  FAIL"

results = []

def check(name, condition, tol=None):
    if tol is not None:
        ok = bool(condition) if not isinstance(condition, (float, np.floating)) else abs(condition) < tol
    else:
        ok = bool(condition)
    results.append(ok)
    print(f"{'  ✅ PASS' if ok else '  ❌ FAIL'}  {name}")
    return ok


# ── shared time axis ─────────────────────────────────────────────────────────
T2     = 5.0
omega0 = 2 * np.pi * 0.5      # 0.5 cycles / µs → easy to check
M0     = 1.0
t      = time_axis(20.0, 0.02)


# ═══════════════════════════════════════════════════════════════════════════
# Group A – initial conditions
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Group A: Initial conditions ──────────────────────────────────────")

Mx_nd, My_nd, Mz_nd = bloch_precession(t, M0=M0, omega0=omega0, T2=None)
check("A1: Mx(0) = M0",    np.isclose(Mx_nd[0], M0))
check("A2: My(0) = 0",     np.isclose(My_nd[0], 0.0, atol=1e-14))
check("A3: Mz(0) = Mz0=0", np.isclose(Mz_nd[0], 0.0))

Mx_d, My_d, Mz_d = bloch_precession(t, M0=M0, omega0=omega0, T2=T2)
check("A4: Mx(0) = M0 with T2",    np.isclose(Mx_d[0], M0))
check("A5: My(0) = 0  with T2",    np.isclose(My_d[0], 0.0, atol=1e-14))

Mx_mz, My_mz, Mz_mz = bloch_precession(t, M0=0.7, omega0=omega0, Mz0=0.5)
check("A6: Mz = Mz0 when Mz0 != 0", np.allclose(Mz_mz, 0.5))


# ═══════════════════════════════════════════════════════════════════════════
# Group B – transverse magnitude invariant (no T2)
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Group B: |M_perp| without T2 ─────────────────────────────────────")

M_perp_nd = np.sqrt(Mx_nd**2 + My_nd**2)
check("B1: |M_perp| constant (max deviation < 1e-12)",
      np.max(np.abs(M_perp_nd - M0)) < 1e-12)
check("B2: |M_perp| = M0 everywhere (allclose)",
      np.allclose(M_perp_nd, M0, rtol=1e-12))
check("B3: magnitude does not drift over 1000 cycles",
      np.allclose(M_perp_nd[-1], M0, rtol=1e-10))


# ═══════════════════════════════════════════════════════════════════════════
# Group C – T2 decay envelope
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Group C: |M_perp| with T2 decay ──────────────────────────────────")

M_perp_d  = np.sqrt(Mx_d**2 + My_d**2)
expected  = M0 * np.exp(-t / T2)

check("C1: |M_perp| matches M0*exp(-t/T2) everywhere",
      np.allclose(M_perp_d, expected, rtol=1e-10))

idx_T2 = np.argmin(np.abs(t - T2))
check("C2: |M_perp|(T2) ≈ 1/e",
      np.isclose(M_perp_d[idx_T2], 1 / np.e, rtol=1e-6))

check("C3: |M_perp| monotonically decreasing",
      np.all(np.diff(M_perp_d) <= 0))

check("C4: |M_perp|(0) = M0",
      np.isclose(M_perp_d[0], M0))

check("C5: |M_perp|(4·T2) ≈ e^-4",
      np.isclose(M_perp_d[np.argmin(np.abs(t - 4*T2))], np.exp(-4), rtol=1e-5))


# ═══════════════════════════════════════════════════════════════════════════
# Group D – Mz behaviour
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Group D: Mz is constant ───────────────────────────────────────────")

check("D1: Mz constant throughout (no T2)",
      np.all(Mz_nd == 0.0))
check("D2: Mz constant throughout (with T2)",
      np.all(Mz_d == 0.0))
check("D3: Mz respects non-zero Mz0",
      np.allclose(Mz_mz, 0.5))


# ═══════════════════════════════════════════════════════════════════════════
# Group E – precession angle
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Group E: Larmor precession angle ─────────────────────────────────")

# check angle at t = pi / omega0  →  Mx should = -M0 (half revolution)
t_half = np.pi / omega0
t_test = np.array([0.0, t_half])
Mx_t, My_t, _ = bloch_precession(t_test, M0=M0, omega0=omega0, T2=None)
check("E1: Mx after half revolution = -M0",
      np.isclose(Mx_t[1], -M0, atol=1e-14))
check("E2: My after quarter revolution = M0",
      np.isclose(
          bloch_precession(np.array([np.pi/(2*omega0)]), M0=M0, omega0=omega0)[1][0],
          M0, atol=1e-14
      ))


# ═══════════════════════════════════════════════════════════════════════════
# Group F – input validation
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Group F: Input validation ─────────────────────────────────────────")

try:
    bloch_precession(t, omega0=-1.0)
    check("F1: raises on omega0 < 0", False)
except ValueError:
    check("F1: raises on omega0 < 0", True)

try:
    bloch_precession(t, T2=0.0)
    check("F2: raises on T2 = 0", False)
except ValueError:
    check("F2: raises on T2 = 0", True)

try:
    bloch_precession(t, T2=-3.0)
    check("F3: raises on T2 < 0", False)
except ValueError:
    check("F3: raises on T2 < 0", True)


# ═══════════════════════════════════════════════════════════════════════════
# Group G – different M0 amplitudes
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Group G: Non-unit M0 ──────────────────────────────────────────────")

for m0 in [0.5, 2.0, 0.01]:
    Mx_g, My_g, _ = bloch_precession(t, M0=m0, omega0=omega0, T2=None)
    mp = np.sqrt(Mx_g**2 + My_g**2)
    check(f"G: |M_perp| = M0={m0} everywhere",
          np.allclose(mp, m0, rtol=1e-12))


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print(f"  {sum(results)}/{len(results)} tests passed")
print(f"{'='*55}")

if __name__ == "__main__":
    pass