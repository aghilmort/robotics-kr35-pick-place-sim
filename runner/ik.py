"""Damped-least-squares (Levenberg-Marquardt style) inverse kinematics for
the 6-DOF arm, solved against a target site pose (position + orientation).

This is a standalone numerical solve run against a scratch copy of qpos --
it does not touch the live simulation. The trajectory planner calls it once
per waypoint to get a joint-space target, then drives the position
actuators toward that target over many physics steps (see trajectory.py).
"""
import mujoco
import numpy as np

from geometry import quat_conj, quat_mul


def _orientation_error(target_quat, current_quat):
    """Axis-angle (3,) error rotating current_quat toward target_quat."""
    dq = quat_mul(target_quat, quat_conj(current_quat))
    if dq[0] < 0:
        dq = -dq
    w = np.clip(dq[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1.0 - w * w, 1e-12))
    if angle < 1e-8:
        return np.zeros(3)
    axis = dq[1:] / s
    return axis * angle


def solve_ik(model, joint_names, site_name, target_pos, target_quat=None,
             q_init=None, max_iters=200, tol_pos=1e-4, tol_rot=np.radians(1.0),
             damping=1e-2, step_scale=1.0):
    """Returns (q_solution, converged, err_pos_norm, err_rot_deg)."""
    data = mujoco.MjData(model)

    dof_idx = []
    qpos_idx = []
    joint_lo = []
    joint_hi = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dof_idx.append(model.jnt_dofadr[jid])
        qpos_idx.append(model.jnt_qposadr[jid])
        joint_lo.append(model.jnt_range[jid, 0])
        joint_hi.append(model.jnt_range[jid, 1])
    dof_idx = np.array(dof_idx)
    qpos_idx = np.array(qpos_idx)
    joint_lo = np.array(joint_lo)
    joint_hi = np.array(joint_hi)

    if q_init is not None:
        data.qpos[qpos_idx] = q_init

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    converged = False
    err_pos_norm = np.inf
    err_rot = 0.0
    for _ in range(max_iters):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

        site_pos = data.site_xpos[site_id].copy()
        site_mat = data.site_xmat[site_id].reshape(3, 3)
        site_quat = np.zeros(4)
        mujoco.mju_mat2Quat(site_quat, site_mat.flatten())

        err_p = target_pos - site_pos
        err_pos_norm = np.linalg.norm(err_p)

        if target_quat is not None:
            err_r = _orientation_error(target_quat, site_quat)
            err_rot = np.linalg.norm(err_r)
            err = np.concatenate([err_p, err_r])
        else:
            err_rot = 0.0
            err = err_p

        if err_pos_norm < tol_pos and err_rot < tol_rot:
            converged = True
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        if target_quat is not None:
            J = np.vstack([jacp[:, dof_idx], jacr[:, dof_idx]])
        else:
            J = jacp[:, dof_idx]

        JJt = J @ J.T + (damping ** 2) * np.eye(J.shape[0])
        dq = J.T @ np.linalg.solve(JJt, err)
        data.qpos[qpos_idx] += step_scale * dq
        data.qpos[qpos_idx] = np.clip(data.qpos[qpos_idx], joint_lo, joint_hi)

    return data.qpos[qpos_idx].copy(), converged, err_pos_norm, np.degrees(err_rot)


def solve_ik_multistart(model, joint_names, site_name, target_pos, target_quat=None,
                         q_init=None, n_restarts=12, seed=0, **kwargs):
    """solve_ik with random-restart fallback.

    The warm-started solve (from q_init, usually the previous waypoint's
    solution) is tried first and is normally sufficient. This 6-DOF arm's
    IK landscape has local minima the damped-least-squares solve can get
    stuck in from a poor seed (elbow/wrist configuration mismatched to the
    target), so on failure we retry from random joint configurations within
    range and keep the best result. Returns the same tuple as solve_ik.
    """
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    lo = np.array([model.jnt_range[j, 0] for j in jids])
    hi = np.array([model.jnt_range[j, 1] for j in jids])
    rng = np.random.default_rng(seed)

    # Try the warm-started seed (q_init, normally the previous waypoint's
    # solution) first, and use it as-is if it converges. A 6-DOF arm hitting
    # a fully-specified pose target usually has only a handful of valid
    # configurations (elbow-up/down, wrist-flip, ...); continuing to search
    # after a good solution is already in hand can swap in a numerically
    # tighter but kinematically distant one (e.g. wrist flipped through the
    # opposite side), producing a huge joint-space jump between waypoints
    # that the arm's position actuators cannot track against gravity. Random
    # restarts are strictly a fallback for when the warm start fails.
    if q_init is not None:
        result = solve_ik(model, joint_names, site_name, target_pos, target_quat, q_init=q_init, **kwargs)
        if result[1]:
            return result

    best = None
    for _ in range(n_restarts):
        q0 = rng.uniform(lo, hi)
        result = solve_ik(model, joint_names, site_name, target_pos, target_quat, q_init=q0, **kwargs)
        _, converged, err_pos, err_rot = result
        score = err_pos + np.radians(err_rot) * 0.01
        if best is None or (converged and not best[1]) or (converged == best[1] and score < best[4]):
            best = (*result, score)
        if converged and err_pos < 1e-4:
            break

    q_sol, converged, err_pos, err_rot, _ = best
    return q_sol, converged, err_pos, err_rot
