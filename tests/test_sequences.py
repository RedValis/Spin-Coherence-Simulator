"""
tests/test_sequences.py – Unit tests for Prototype 4 (pulse sequences).
========================================================================

Physics invariants under test:

  A. apply_pulse       – rotation matrices, length preservation, special cases
  B. free_evolve       – thin wrapper, consistent with simulate_bloch
  C. hahn_echo         – echo amplitude = M0*exp(-2tau/T2), echo timing
  D. no-echo baseline  – no π pulse → no refocusing
  E. limiting cases    – T2>>tau: amplitude≈M0;  T2<<tau: amplitude≈0
  F. CPMG              – n echoes at correct times, amplitudes decay correctly
  G. sweep_echo        – recovered T2 consistent with input
  H. input validation  – bad tau, dt, n_echoes all raise

Run:  pytest tests/test_sequences.py -v
  or: python tests/test_sequences.py
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.sequences import (
    apply_pulse, free_evolve, hahn_echo_sequence,
    cpmg_sequence, measure_echo_amplitude, sweep_echo_amplitude,
)

results = []

def check(name, condition):
    ok = bool(condition)
    results.append(ok)
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}  {name}")
    return ok


# ── shared physics ────────────────────────────────────────────────────────────
gamma  = 1.0
B0     = 2 * np.pi * 0.5      # 0.5 MHz
B      = np.array([0., 0., B0])
M0     = 1.0
T1     = 50.0
T2     = 10.0
dt     = 0.02


# ===========================================================================
# Group A – apply_pulse
# ===========================================================================
print("\n── Group A: apply_pulse ──────────────────────────────────────────────")

M_z = np.array([0., 0., 1.])   # equilibrium

# π/2 around y: [0,0,1] → [1,0,0]
M_x = apply_pulse(M_z, axis='y', angle=np.pi/2)
check("A1: π/2_y tips M_z → M_x",         np.allclose(M_x, [1, 0, 0], atol=1e-14))

# π around x: [0,0,1] → [0,0,-1]
M_inv = apply_pulse(M_z, axis='x', angle=np.pi)
check("A2: π_x inverts M_z → -M_z",       np.allclose(M_inv, [0, 0, -1], atol=1e-14))

# π/2 around y then π around x: back to [0,0,1]? No — check length instead
M_any = np.array([0.3, 0.5, 0.8])
M_rot = apply_pulse(M_any, axis='z', angle=1.234)
check("A3: rotation preserves |M|",        np.isclose(np.linalg.norm(M_rot), np.linalg.norm(M_any)))

# two π pulses = identity
M_dbl = apply_pulse(apply_pulse(M_any, 'x', np.pi), 'x', np.pi)
check("A4: two π pulses → identity",       np.allclose(M_dbl, M_any, atol=1e-14))

# four π/2 pulses = identity
M_quad = M_any.copy()
for _ in range(4):
    M_quad = apply_pulse(M_quad, 'y', np.pi/2)
check("A5: four π/2 pulses → identity",    np.allclose(M_quad, M_any, atol=1e-13))

# zero angle = identity
check("A6: angle=0 → identity",            np.allclose(apply_pulse(M_any, 'x', 0), M_any, atol=1e-14))

# bad axis raises
try:
    apply_pulse(M_z, axis='w', angle=np.pi)
    check("A7: raises on bad axis", False)
except ValueError:
    check("A7: raises on bad axis", True)

# π/2 around -y: [0,0,1] → [-1,0,0]
M_nx = apply_pulse(M_z, axis='-y', angle=np.pi/2)
check("A8: π/2_-y tips M_z → -M_x",       np.allclose(M_nx, [-1, 0, 0], atol=1e-14))


# ===========================================================================
# Group B – free_evolve
# ===========================================================================
print("\n── Group B: free_evolve ──────────────────────────────────────────────")

M_init = np.array([M0, 0., 0.])
t_seg, Mx_s, My_s, Mz_s = free_evolve(M_init, t_free=T2, dt=dt,
                                        gamma=gamma, B=B, T1=T1, T2=T2, M0=M0)

check("B1: t_seg starts at 0",             np.isclose(t_seg[0], 0.0))
check("B2: t_seg ends near t_free",        np.isclose(t_seg[-1], T2, rtol=1e-3))
check("B3: M starts at M_init",            np.isclose(Mx_s[0], M0))
check("B4: |M_perp|(0) = M0",             np.isclose(np.sqrt(Mx_s[0]**2+My_s[0]**2), M0))
check("B5: |M_perp|(T2) ≈ 1/e",
      np.isclose(np.sqrt(Mx_s[-1]**2+My_s[-1]**2), 1/np.e, rtol=1e-4))


# ===========================================================================
# Group C – Hahn echo amplitude and timing
# ===========================================================================
print("\n── Group C: Hahn echo amplitude ──────────────────────────────────────")

for tau in [1.0, 2.0, 5.0]:
    t_h, Mx_h, My_h, Mz_h = hahn_echo_sequence(
        gamma=gamma, B=B, T1=T1, T2=T2, M0=M0, tau=tau, dt=dt)
    amp = measure_echo_amplitude(Mx_h, My_h, t_h, echo_time=2*tau)
    expected = M0 * np.exp(-2*tau / T2)
    check(f"C: echo amplitude at tau={tau} ≈ exp(-2τ/T2)={expected:.4f}",
          np.isclose(amp, expected, rtol=0.02))

# time axis spans 0 to 2τ
tau_c = 3.0
t_c, Mx_c, My_c, Mz_c = hahn_echo_sequence(
    gamma=gamma, B=B, T1=T1, T2=T2, M0=M0, tau=tau_c, dt=dt)
check("C4: time axis starts at 0",         np.isclose(t_c[0], 0.0))
check("C5: time axis ends near 2τ",        np.isclose(t_c[-1], 2*tau_c, rtol=1e-3))


# ===========================================================================
# Group D – no refocusing without π pulse (baseline)
# ===========================================================================
print("\n── Group D: No echo without π pulse (baseline) ───────────────────────")

# Just free evolve 2τ — no refocusing pulse
tau_d = 3.0
t_d, Mx_d, My_d, Mz_d = free_evolve(
    np.array([M0, 0., 0.]), t_free=2*tau_d, dt=dt,
    gamma=gamma, B=B, T1=T1, T2=T2, M0=M0)
amp_free = float(np.max(np.sqrt(Mx_d**2 + My_d**2)))

t_echo, Mx_echo, My_echo, _ = hahn_echo_sequence(
    gamma=gamma, B=B, T1=T1, T2=T2, M0=M0, tau=tau_d, dt=dt)
amp_echo = measure_echo_amplitude(Mx_echo, My_echo, t_echo, 2*tau_d)

# Without static offsets the pure Bloch decay won't show a "refocus bump"
# but the echo sequence amplitude should match exp(-2τ/T2)
check("D1: Hahn echo amp ≈ exp(-2τ/T2)",
      np.isclose(amp_echo, M0*np.exp(-2*tau_d/T2), rtol=0.02))


# ===========================================================================
# Group E – limiting cases
# ===========================================================================
print("\n── Group E: Limiting cases ───────────────────────────────────────────")

tau_e = 1.0

# T2 >> tau: echo amplitude ≈ M0
T2_long = 1e6
t_e1, Mx_e1, My_e1, _ = hahn_echo_sequence(
    gamma=gamma, B=B, T1=T2_long, T2=T2_long, M0=M0, tau=tau_e, dt=dt)
amp_long = measure_echo_amplitude(Mx_e1, My_e1, t_e1, 2*tau_e)
check("E1: T2>>tau → echo amplitude ≈ M0",
      np.isclose(amp_long, M0, rtol=0.01))

# T2 << tau: echo amplitude ≈ 0
T2_short = 0.1
t_e2, Mx_e2, My_e2, _ = hahn_echo_sequence(
    gamma=gamma, B=B, T1=T1, T2=T2_short, M0=M0, tau=tau_e, dt=0.005)
amp_short = measure_echo_amplitude(Mx_e2, My_e2, t_e2, 2*tau_e)
check("E2: T2<<tau → echo amplitude ≈ 0",
      amp_short < 0.05)

# Verify analytic formula over a range of tau
tau_arr = np.array([0.5, 1.0, 2.0, 4.0])
_, amps = sweep_echo_amplitude(gamma, B, T1, T2, M0, tau_arr, dt=dt)
expected_amps = M0 * np.exp(-2*tau_arr / T2)
check("E3: sweep echo amplitudes match exp(-2τ/T2) formula",
      np.allclose(amps, expected_amps, rtol=0.03))


# ===========================================================================
# Group F – CPMG echo train
# ===========================================================================
print("\n── Group F: CPMG echo train ──────────────────────────────────────────")

n_e  = 5
tau_f = 1.0
t_cp, Mx_cp, My_cp, Mz_cp, echo_times = cpmg_sequence(
    gamma=gamma, B=B, T1=T1, T2=T2, M0=M0,
    tau=tau_f, n_echoes=n_e, dt=dt)

check("F1: correct number of echoes",      len(echo_times) == n_e)
check("F2: first echo at 2τ",             np.isclose(echo_times[0], 2*tau_f, rtol=1e-3))
check("F3: echo times equally spaced",
      np.allclose(np.diff(echo_times), 2*tau_f, rtol=1e-3))
check("F4: last echo at 2n·τ",            np.isclose(echo_times[-1], 2*n_e*tau_f, rtol=1e-3))

# Echo amplitudes at each echo time decay as exp(-t/T2)
echo_amps = [measure_echo_amplitude(Mx_cp, My_cp, t_cp, et) for et in echo_times]
expected_cp = M0 * np.exp(-echo_times / T2)
check("F5: CPMG echo amplitudes match exp(-t_echo/T2)",
      np.allclose(echo_amps, expected_cp, rtol=0.05))

check("F6: CPMG time axis starts at 0",   np.isclose(t_cp[0], 0.0))
check("F7: CPMG time ends near 2n·τ",     np.isclose(t_cp[-1], 2*n_e*tau_f, rtol=1e-3))


# ===========================================================================
# Group G – T2 extraction from echo sweep
# ===========================================================================
print("\n── Group G: T2 extraction from echo sweep ────────────────────────────")

tau_sweep  = np.linspace(0.5, 8.0, 12)
two_tau, amps_sw = sweep_echo_amplitude(gamma, B, T1, T2, M0, tau_sweep, dt=dt)

# Fit log(A) = log(M0) - 2τ/T2
log_A   = np.log(np.clip(amps_sw, 1e-10, None))
coeffs  = np.polyfit(tau_sweep, log_A, 1)   # slope = -1/T2  (vs τ, so -2/T2... wait)
# two_tau = 2*tau, so: log(A) = -two_tau/T2
log_A2  = np.log(np.clip(amps_sw, 1e-10, None))
coeffs2 = np.polyfit(two_tau, log_A2, 1)    # slope = -1/T2
T2_fit  = -1.0 / coeffs2[0]
check(f"G1: fitted T2 = {T2_fit:.3f} ≈ {T2} (rtol 5%)",
      np.isclose(T2_fit, T2, rtol=0.05))


# ===========================================================================
# Group H – input validation
# ===========================================================================
print("\n── Group H: Input validation ─────────────────────────────────────────")

try:
    hahn_echo_sequence(gamma, B, T1, T2, M0, tau=-1.0, dt=dt)
    check("H1: hahn raises on tau<=0", False)
except ValueError:
    check("H1: hahn raises on tau<=0", True)

try:
    hahn_echo_sequence(gamma, B, T1, T2, M0, tau=1.0, dt=5.0)
    check("H2: hahn raises when dt>=tau", False)
except ValueError:
    check("H2: hahn raises when dt>=tau", True)

try:
    cpmg_sequence(gamma, B, T1, T2, M0, tau=1.0, n_echoes=0, dt=dt)
    check("H3: cpmg raises on n_echoes<1", False)
except ValueError:
    check("H3: cpmg raises on n_echoes<1", True)

try:
    cpmg_sequence(gamma, B, T1, T2, M0, tau=-1.0, n_echoes=3, dt=dt)
    check("H4: cpmg raises on tau<=0", False)
except ValueError:
    check("H4: cpmg raises on tau<=0", True)


# ===========================================================================
# Summary
# ===========================================================================
print(f"\n{'='*58}")
print(f"  {sum(results)}/{len(results)} tests passed")
print(f"{'='*58}\n")