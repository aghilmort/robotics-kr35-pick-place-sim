#!/usr/bin/env python3
"""
Empirical settle-test used to validate (not guess) the position actuator
gains (kp/kv) in build_kr35_mjcf.py against KR35's real per-link masses
(243kg total vs. the earlier KR16 build's placeholder ~90kg -- see
LINK_INERTIALS there). Holds a set of representative poses under gravity for
8s of simulated time each with ctrl pinned at the starting qpos, and reports
the residual position error once velocities settle to zero. A few degrees of
sag under a 400000/20000 vs. this real mass distribution would mean the
gains are too soft; sub-0.15-degree residuals across outstretched, loaded,
and twisted poses is the bar this was checked against.

Run: cd robot && MUJOCO_GL=egl python3 tune_gains.py
"""
import mujoco
import numpy as np
import pathlib

TEST_POSES = {
    "outstretched (zero pose, max shoulder moment arm)": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "elbow-up load": np.array([0.0, -1.0, 2.5, 0.0, -1.0, 0.0]),
    "twisted (all joints off-axis)": np.array([1.0, -0.8, 1.5, 1.5, 1.0, 2.0]),
}


def main():
    xml_path = pathlib.Path(__file__).parent / "kr35_r1840_3_hw.xml"
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    d = mujoco.MjData(m)

    print(f"{'pose':40s} {'max err (deg)':>14s} {'max qvel':>10s}")
    for label, q in TEST_POSES.items():
        d.qpos[:] = q
        d.ctrl[:] = q
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        for _ in range(4000):  # 8s simulated at the model's 2ms timestep
            mujoco.mj_step(m, d)
        err_deg = np.degrees(np.abs(d.qpos - q)).max()
        print(f"{label:40s} {err_deg:14.3f} {np.abs(d.qvel).max():10.4f}")


if __name__ == "__main__":
    main()
