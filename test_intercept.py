"""Verify the secant-method intercept fix across parameter ranges.

Compares against a reference bisection solver (ground truth) to ensure
the new secant method converges to the correct intercept angle for all
cases — including the previously-divergent regime (ω·r / v > 1).
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl.action import compute_intercept_angle

_SUN_CENTER = (50.0, 50.0)


def reference_intercept(src_x, src_y, r, theta_0, angular_velocity, fleet_speed):
    """Bisection solver (ground truth) for the intercept problem.

    Solves  f(t) = distance(src, target(t)) - fleet_speed * t = 0
    """

    def target_pos(t):
        th = theta_0 + angular_velocity * t
        return (
            _SUN_CENTER[0] + r * math.cos(th),
            _SUN_CENTER[1] + r * math.sin(th),
        )

    def f(t):
        tx, ty = target_pos(t)
        d = math.hypot(ty - src_y, tx - src_x)
        return d - fleet_speed * t

    # Find bracket [lo, hi] where f changes sign
    lo, hi = 0.0, 1.0
    # Expand hi until f(hi) <= 0 or hi is huge
    for _ in range(20):
        if f(hi) <= 0:
            break
        hi *= 2.0
    else:
        # Never caught up — fleet too slow; return closest-approach aim
        t = lo
        tx, ty = target_pos(t)
        return math.atan2(ty - src_y, tx - src_x), tx, ty

    # Bisection
    for _ in range(60):
        mid = (lo + hi) / 2.0
        fm = f(mid)
        if abs(fm) < 1e-6:
            lo = hi = mid
            break
        if fm > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-8:
            break

    t = (lo + hi) / 2.0
    tx, ty = target_pos(t)
    angle = math.atan2(ty - src_y, tx - src_x)
    return angle, tx, ty


def make_orbit_lookup(tgt_id, r, initial_angle, rotates=True):
    return {
        tgt_id: {
            'orbital_r': r,
            'initial_angle': initial_angle,
            'rotates': rotates,
        }
    }


def test_case(name, src_x, src_y, tgt_r, tgt_angle_0, angular_velocity,
              fleet_speed, tgt_id=99):
    """Run one test case and report results."""
    cx, cy = _SUN_CENTER
    tgt_x = cx + tgt_r * math.cos(tgt_angle_0)
    tgt_y = cy + tgt_r * math.sin(tgt_angle_0)

    lookup = make_orbit_lookup(tgt_id, tgt_r, tgt_angle_0)

    # New secant method
    angle_new, ftx_new, fty_new = compute_intercept_angle(
        src_x=src_x, src_y=src_y,
        tgt_id=tgt_id, tgt_x=tgt_x, tgt_y=tgt_y,
        fleet_speed=fleet_speed,
        orbit_lookup=lookup,
        angular_velocity=angular_velocity,
        return_pos=True,
    )

    # Reference bisection
    angle_ref, ftx_ref, fty_ref = reference_intercept(
        src_x, src_y, tgt_r, tgt_angle_0, angular_velocity, fleet_speed,
    )

    angle_err = abs(angle_new - angle_ref)
    pos_err = math.hypot(ftx_new - ftx_ref, fty_new - fty_ref)

    omega_r_v = angular_velocity * tgt_r / max(fleet_speed, 0.01)

    status = "✅" if angle_err < 0.01 and pos_err < 1.0 else "❌"
    print(f"{status} {name}")
    print(f"   ω·r/v={omega_r_v:.3f}  angle_err={angle_err:.6f}rad  pos_err={pos_err:.4f}")
    if angle_err >= 0.01:
        print(f"   NEW:  angle={angle_new:.4f}  target=({ftx_new:.2f},{fty_new:.2f})")
        print(f"   REF:  angle={angle_ref:.4f}  target=({ftx_ref:.2f},{fty_ref:.2f})")

    return angle_err < 0.01


def main():
    print("=" * 60)
    print("Intercept Angle Convergence Tests (Secant Method)")
    print("=" * 60)

    all_pass = True

    # ── Case 1: previously divergent (ω·r/v = 2.0) ─────────────────
    all_pass &= test_case(
        "Divergent case: slow fleet, fast far planet (ω·r/v=2.0)",
        src_x=30.0, src_y=70.0,      # source in top-left quadrant
        tgt_r=40.0, tgt_angle_0=0.0, # target at (90, 50) on far orbit
        angular_velocity=0.05,        # max ω
        fleet_speed=1.0,              # min speed (1 ship)
    )

    # ── Case 2: borderline (ω·r/v ≈ 1.0) ──────────────────────────
    all_pass &= test_case(
        "Borderline: ω·r/v ≈ 1.0",
        src_x=40.0, src_y=60.0,
        tgt_r=33.0, tgt_angle_0=0.5,
        angular_velocity=0.04,
        fleet_speed=1.32,
    )

    # ── Case 3: well-behaved (ω·r/v = 0.15) ───────────────────────
    all_pass &= test_case(
        "Well-behaved: fast fleet, close planet (ω·r/v=0.15)",
        src_x=35.0, src_y=65.0,
        tgt_r=20.0, tgt_angle_0=1.0,
        angular_velocity=0.03,
        fleet_speed=4.0,
    )

    # ── Case 4: source close to target ─────────────────────────────
    all_pass &= test_case(
        "Nearby target",
        src_x=55.0, src_y=52.0,
        tgt_r=10.0, tgt_angle_0=0.3,
        angular_velocity=0.04,
        fleet_speed=3.0,
    )

    # ── Case 5: source at center (degenerate geometry) ─────────────
    all_pass &= test_case(
        "Source near center",
        src_x=50.0, src_y=50.0,
        tgt_r=30.0, tgt_angle_0=2.0,
        angular_velocity=0.04,
        fleet_speed=2.0,
    )

    # ── Case 6: extreme divergence (ω·r/v = 2.5) ──────────────────
    all_pass &= test_case(
        "Extreme: very slow fleet vs very fast far planet (ω·r/v=2.5)",
        src_x=20.0, src_y=80.0,
        tgt_r=40.0, tgt_angle_0=3.0,
        angular_velocity=0.05,
        fleet_speed=0.8,  # very slow (unusual but possible)
    )

    # ── Case 7: static planet (no rotation) ────────────────────────
    lookup_static = {
        99: {'orbital_r': 30.0, 'initial_angle': 0.5, 'rotates': False},
    }
    angle_s, ftx_s, fty_s = compute_intercept_angle(
        src_x=30.0, src_y=70.0, tgt_id=99,
        tgt_x=75.0, tgt_y=55.0, fleet_speed=2.0,
        orbit_lookup=lookup_static, angular_velocity=0.04,
        return_pos=True,
    )
    ref_angle = math.atan2(55.0 - 70.0, 75.0 - 30.0)
    static_ok = abs(angle_s - ref_angle) < 1e-6
    print(f"{'✅' if static_ok else '❌'} Static planet (no rotation)")
    all_pass &= static_ok

    # ── Summary ────────────────────────────────────────────────────
    print()
    print("=" * 60)
    if all_pass:
        print("ALL TESTS PASSED ✅")
    else:
        print("SOME TESTS FAILED ❌")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
