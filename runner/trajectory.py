"""Scripted pick-and-place execution: waypoint generation, joint-space
interpolated motion, contact-triggered suction engage/release, and
success-criteria evaluation.

The overall episode is a fixed state machine:
  pre_settle -> pick_hover -> pick_descend(+engage) -> pick_lift ->
  place_hover -> place_descend(+release) -> retreat -> final_settle

Every phase consumes from one shared sim-time budget
(success_criteria.max_sim_time_s); running out mid-episode is reported as a
timeout failure.
"""
import sys
import pathlib

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "robot"))

import ik
import planner
from geometry import compose, invert, relative_pose, quat_from_z_align, quat_angle_deg, euler_deg_to_quat
from build_kr35_mjcf import JOINTS as ROBOT_JOINT_SPECS, SUCTION_SITE_LOCAL_POS

MAX_JOINT_VEL = np.array([j.velocity for j in ROBOT_JOINT_SPECS])  # rad/s, from real KUKA spec

# The suction weld equality constraint's body1 is "suction_cup" -- the BODY,
# not the "suction_site" the IK solver targets -- so its relpose (eq_data)
# must be captured/expressed in the body's frame. Everywhere we need the
# site's pose instead (IK targets), convert with this fixed local offset.
_SITE_LOCAL_POS = np.array(SUCTION_SITE_LOCAL_POS)
_SITE_LOCAL_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

# Tuned for this arm's 6-DOF damped-least-squares IK: tight enough that a
# 2cm/20deg-class success tolerance is never at risk, loose enough to
# converge reliably within a handful of random restarts instead of needing
# hundreds of iterations chasing the last fraction of a millimeter.
IK_KWARGS = dict(n_restarts=20, max_iters=400, tol_pos=1e-3, tol_rot=np.radians(2.0),
                  damping=5e-3, step_scale=0.7)
_IK_SOLVE_KWARGS = {k: v for k, v in IK_KWARGS.items() if k != "n_restarts"}


# Normal (non-detour) transit reconfigurations in this arm's episodes -- the
# ordinary "pick_lift config -> place_hover config" jump with no obstacle
# forcing a branch change -- measured 1.18-1.69 rad L2 across 29 sampled
# trials (basic_pick_place.yaml + obstacle_pick_place.yaml's non-pathological
# seeds), median ~1.29 rad. MAX_GOAL_JUMP is set well above that range (to
# leave real room for a legitimate obstacle-driven detour) but well below
# the ~5-8.6 rad jumps a full-joint-range random restart can land on when it
# finds a different-but-collision-free IK branch -- e.g. a near-full-turn
# wrist flip, ~145-260 degrees on a single joint. Those are numerically
# valid and genuinely collision-free at the endpoint, but sweep through such
# a different overall arm configuration to get there that they reliably
# flung or dropped the held object well before arriving in testing (see
# kr16_visual_fidelity_notes.md / the planner section for the measured
# failure trace). Rejecting them outright and failing fast with a clear
# reason is better than a slow-motion failure that burns the whole episode's
# sim-time budget arriving at a technically-valid but unusable configuration.
MAX_GOAL_JUMP = 2.5  # rad, L2 over all 6 joints


def _search_collision_free_ik(model, ctx, target_pos, target_quat, checker, q_cur,
                               n_restarts=60, max_jump=MAX_GOAL_JUMP, seed=0):
    """Bounded random-restart search for an IK solution to (target_pos,
    target_quat) that the given CollisionChecker judges collision-free AND
    within max_jump of q_cur in joint space -- used only as a fallback when
    the normal warm-started solve converges to a colliding branch (see the
    call site in run_episode).

    Deliberately does NOT stop at the first collision-free hit: it evaluates
    the full restart budget and keeps the closest-to-q_cur candidate that
    also clears the jump cap, same "prefer proximity over a numerically-
    nicer but kinematically-distant solution" philosophy
    ik.solve_ik_multistart already uses for its own warm start -- see
    MAX_GOAL_JUMP above for why the cap exists at all.

    Returns (q, converged, err_pos, err_rot); on total failure,
    converged=False and err_pos=inf so the caller can report a clear reason
    rather than silently reusing a colliding or wildly-distant solution.
    """
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ctx.joint_names]
    lo = np.array([model.jnt_range[j, 0] for j in jids])
    hi = np.array([model.jnt_range[j, 1] for j in jids])
    rng = np.random.default_rng(seed)
    best = None  # (dist, q, err_pos, err_rot)
    for _ in range(n_restarts):
        q0 = rng.uniform(lo, hi)
        q, converged, err_pos, err_rot = ik.solve_ik(model, ctx.joint_names, ctx.suction_site,
                                                       target_pos, target_quat, q_init=q0, **_IK_SOLVE_KWARGS)
        if not converged or checker.in_collision(q):
            continue
        dist = float(np.linalg.norm(q - q_cur))
        if dist > max_jump:
            continue
        if best is None or dist < best[0]:
            best = (dist, q, err_pos, err_rot)
    if best is None:
        return None, False, np.inf, 0.0
    _, q, err_pos, err_rot = best
    return q, True, err_pos, err_rot


class Budget:
    def __init__(self, total_s):
        self.remaining = total_s

    def spend(self, dt_s):
        self.remaining -= dt_s
        return self.remaining > 0


def _top_offset_local(shape, size):
    if shape == "box":
        return np.array([0.0, 0.0, size[2]])
    if shape == "cylinder":
        return np.array([0.0, 0.0, size[1]])
    if shape == "sphere":
        return np.array([0.0, 0.0, size[0]])
    raise ValueError(shape)


def _body_pose(data, body_id):
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.xmat[body_id])
    return data.xpos[body_id].copy(), quat


def _site_pose(data, site_id):
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
    return data.site_xpos[site_id].copy(), quat


def _qpos_idx(model, joint_names):
    return [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joint_names]


def _actuator_idx(model, joint_names):
    # joint_names are "robot_joint_1" etc (prefixed by MjSpec.attach); the
    # matching actuator, defined as "act_joint_1" in the robot-only MJCF,
    # gets the SAME "robot_" prefix applied by attach -> "robot_act_joint_1".
    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"robot_act_{n[len('robot_'):]}") for n in joint_names]


def _step_hold(model, data, ctrl_idx, ctrl_val, n_steps, budget, log=None):
    """Step physics n_steps times holding actuators at ctrl_val."""
    for _ in range(n_steps):
        if not budget.spend(model.opt.timestep):
            return False
        data.ctrl[ctrl_idx] = ctrl_val
        mujoco.mj_step(model, data)
        if log is not None:
            log(data)
    return True


def _move_to(model, data, ctx, q_target, budget, speed_scale, contact_check=None, log=None):
    """Linearly interpolate the arm's actuator targets from current qpos to
    q_target, timed from the real per-joint max velocities. Steps physics
    the whole way. If contact_check is given, it's called every step with
    (model, data) and may return True to end the motion early (used for the
    pick-descent contact stop).

    Returns "ok", "contact", or "timeout".
    """
    qpos_idx = _qpos_idx(model, ctx.joint_names)
    ctrl_idx = _actuator_idx(model, ctx.joint_names)
    q_start = data.qpos[qpos_idx].copy()
    dq = q_target - q_start
    t_needed = np.max(np.abs(dq) / (MAX_JOINT_VEL * max(speed_scale, 1e-3)))
    t_needed = max(t_needed, 0.05)
    n_steps = max(1, int(np.ceil(t_needed / model.opt.timestep)))

    for step in range(1, n_steps + 1):
        if not budget.spend(model.opt.timestep):
            return "timeout"
        alpha = step / n_steps
        # smoothstep for gentler accel/decel than pure linear
        alpha_s = 3 * alpha ** 2 - 2 * alpha ** 3
        data.ctrl[ctrl_idx] = q_start + dq * alpha_s
        mujoco.mj_step(model, data)
        if log is not None:
            log(data)
        if contact_check is not None and contact_check(model, data):
            return "contact"
    return "ok"


def _find_contact(model, data, body_a, body_b):
    for i in range(data.ncon):
        c = data.contact[i]
        ba = model.geom_bodyid[c.geom1]
        bb = model.geom_bodyid[c.geom2]
        if {ba, bb} == {body_a, body_b}:
            return c
    return None


def _compute_pick_targets(scenario, ctx, model, data):
    task = scenario["task"]["pick"]
    obj_id = task["object_id"]
    info = ctx.objects[obj_id]
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, info.body_name)
    obj_pos, obj_quat = _body_pose(data, obj_body)

    if "grasp_point" in task:
        local_pt = np.array(task["grasp_point"], dtype=float)
    else:
        local_pt = _top_offset_local(info.shape, info.size)
    approach_axis = np.array(task.get("approach_axis", [0, 0, -1]), dtype=float)
    approach_axis /= np.linalg.norm(approach_axis)

    grasp_pos, _ = compose(obj_pos, obj_quat, local_pt, np.array([1.0, 0, 0, 0]))
    grasp_quat = quat_from_z_align(approach_axis)
    return grasp_pos, grasp_quat, approach_axis, obj_body, info


def _compute_place_object_target(scenario, ctx):
    task = scenario["task"]["place"]
    if "target_pose" in task:
        pos = np.array(task["target_pose"].get("pos", [0, 0, 0]), dtype=float)
        rpy = task["target_pose"].get("rpy_deg", [0, 0, 0])
        return pos, euler_deg_to_quat(rpy)
    fixture_id = task["target_fixture_id"]
    top_pos, top_yaw = ctx.fixture_top[fixture_id]
    obj_id = scenario["task"]["pick"]["object_id"]
    info = ctx.objects[obj_id]
    half_h = _top_offset_local(info.shape, info.size)[2]
    pos = top_pos + np.array([0, 0, half_h])
    quat = euler_deg_to_quat([0, 0, np.degrees(top_yaw)])
    return pos, quat


def run_episode(model, data, ctx, scenario, on_frame=None):
    """Executes the full pick-and-place episode. Returns a report dict."""
    sc = scenario["success_criteria"]
    motion = scenario["task"].get("motion", {})
    approach_h = motion.get("approach_height_m", 0.15)
    speed_scale = motion.get("joint_speed_scale", 0.5)

    budget = Budget(sc["max_sim_time_s"])
    log = (lambda d: on_frame(d)) if on_frame else None
    report = {"name": scenario["name"], "phases": [], "success": False, "failure_reason": None}

    qpos_idx = _qpos_idx(model, ctx.joint_names)
    ctrl_idx = _actuator_idx(model, ctx.joint_names)
    suction_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ctx.suction_site)
    suction_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ctx.suction_cup_body)

    # --- phase 0: let the scene settle physically before planning ---
    home_ctrl = data.qpos[qpos_idx].copy()
    if not _step_hold(model, data, ctrl_idx, home_ctrl, 150, budget, log):
        report["failure_reason"] = "timeout_during_presettle"
        return report
    report["phases"].append("pre_settle")

    pick_pos, pick_quat, approach_axis, obj_body, pick_info = _compute_pick_targets(scenario, ctx, model, data)
    if pick_info.eq_id is None or pick_info.eq_id < 0:
        report["failure_reason"] = "pick_object_not_graspable"
        return report

    pick_hover = pick_pos - approach_axis * approach_h

    q_cur = data.qpos[qpos_idx].copy()

    # --- phase 1: hover above pick point ---
    q_target, conv, ep, er = ik.solve_ik_multistart(model, ctx.joint_names, ctx.suction_site,
                                                      pick_hover, pick_quat, q_init=q_cur, **IK_KWARGS)
    if not conv:
        report["failure_reason"] = f"ik_failed_pick_hover (pos_err={ep:.4f}, rot_err={er:.1f}deg)"
        return report
    status = _move_to(model, data, ctx, q_target, budget, speed_scale, log=log)
    if status != "ok":
        report["failure_reason"] = f"move_failed_pick_hover ({status})"
        return report
    report["phases"].append("pick_hover")
    q_cur = q_target

    # --- phase 2: descend to contact ---
    # Target a point slightly PAST the analytic surface point (probe_overshoot,
    # along approach_axis) rather than the exact surface: sub-millimeter IK
    # residual error means aiming exactly at the surface often leaves a gap
    # too small to ever generate an actual geometric contact. contact_check
    # halts the motion the instant contact fires, so the overshoot is a
    # safety margin, not something the cup actually travels through.
    probe_overshoot = 0.03
    pick_probe_target = pick_pos + approach_axis * probe_overshoot
    q_target, conv, ep, er = ik.solve_ik_multistart(model, ctx.joint_names, ctx.suction_site,
                                                      pick_probe_target, pick_quat, q_init=q_cur, **IK_KWARGS)
    if not conv:
        report["failure_reason"] = f"ik_failed_pick_contact (pos_err={ep:.4f}, rot_err={er:.1f}deg)"
        return report

    def check_pick_contact(m, d):
        return _find_contact(m, d, suction_body, obj_body) is not None

    status = _move_to(model, data, ctx, q_target, budget, speed_scale * 0.15, contact_check=check_pick_contact, log=log)
    if status == "timeout":
        report["failure_reason"] = "timeout_during_pick_descend"
        return report
    report["phases"].append("pick_descend")

    if status != "contact":
        # reached the analytic contact point without a registered contact
        # (e.g. object shifted slightly) -- fail fast rather than crush it
        if _find_contact(model, data, suction_body, obj_body) is None:
            report["failure_reason"] = "no_contact_at_pick_point"
            return report

    # --- engage suction: capture live relative transform, arm the weld ---
    # The weld's body1 is the suction_cup BODY (not the site), so relpose
    # must be captured in the body's frame -- see _SITE_LOCAL_POS comment.
    cup_body_pos, cup_body_quat = _body_pose(data, suction_body)
    obj_pos, obj_quat = _body_pose(data, obj_body)
    rel_pos, rel_quat = relative_pose(cup_body_pos, cup_body_quat, obj_pos, obj_quat)

    max_payload = scenario["robot"].get("suction", {}).get("max_payload_kg")
    if max_payload is not None and pick_info.eq_id is not None:
        obj_mass = model.body_mass[obj_body]
        if obj_mass > max_payload:
            report["failure_reason"] = f"suction_overload (object {obj_mass:.2f}kg > max {max_payload:.2f}kg)"
            return report

    eq_id = pick_info.eq_id
    model.eq_data[eq_id, 0:3] = 0.0
    model.eq_data[eq_id, 3:6] = rel_pos
    model.eq_data[eq_id, 6:10] = rel_quat
    model.eq_data[eq_id, 10] = 1.0
    data.eq_active[eq_id] = 1
    report["phases"].append("suction_engaged")
    q_cur = data.qpos[qpos_idx].copy()

    # --- phase 3: lift back to hover ---
    q_target, conv, ep, er = ik.solve_ik_multistart(model, ctx.joint_names, ctx.suction_site,
                                                      pick_hover, pick_quat, q_init=q_cur, **IK_KWARGS)
    if not conv:
        report["failure_reason"] = f"ik_failed_pick_lift (pos_err={ep:.4f}, rot_err={er:.1f}deg)"
        return report
    status = _move_to(model, data, ctx, q_target, budget, speed_scale, log=log)
    if status != "ok":
        report["failure_reason"] = f"move_failed_pick_lift ({status})"
        return report
    report["phases"].append("pick_lift")
    q_cur = q_target

    # --- compute place targets from the just-captured grasp transform ---
    # Solve for the required suction_cup BODY pose (object_target = cup ∘
    # rel), then convert that to a suction_site pose since that's what the
    # IK solver (and _move_to waypoints) actually target.
    obj_target_pos, obj_target_quat = _compute_place_object_target(scenario, ctx)
    cup_body_target_pos, cup_body_target_quat = compose(obj_target_pos, obj_target_quat, *invert(rel_pos, rel_quat))
    place_pos, place_quat = compose(cup_body_target_pos, cup_body_target_quat, _SITE_LOCAL_POS, _SITE_LOCAL_QUAT)
    place_hover = place_pos - approach_axis * approach_h

    # --- phase 3b: obstacle-aware transit (joint-space RRT-Connect) ---
    # This is the one part of the episode where something sitting between
    # the pick and place locations can actually matter, so it's the one
    # part that goes through a real motion planner instead of a hand-picked
    # waypoint. (The short descend/lift moves elsewhere stay plain
    # interpolation -- there's no useful planning problem in a 5cm
    # straight-down move next to a known target.) See planner.py.
    q_target, conv, ep, er = ik.solve_ik_multistart(model, ctx.joint_names, ctx.suction_site,
                                                      place_hover, place_quat, q_init=q_cur, **IK_KWARGS)
    if not conv:
        report["failure_reason"] = f"ik_failed_place_hover (pos_err={ep:.4f}, rot_err={er:.1f}deg)"
        return report

    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ctx.joint_names]
    joint_lo = np.array([model.jnt_range[j, 0] for j in jids])
    joint_hi = np.array([model.jnt_range[j, 1] for j in jids])
    # The picked object moves rigidly with the suction cup for the whole
    # transit (that's what the weld constraint does at runtime); the
    # planner needs to know that so it can predict the object's swept
    # volume at each candidate arm configuration, not just the arm's own.
    obj_qpos_adr = model.jnt_qposadr[model.body_jntadr[obj_body]]
    held = {"cup_body_id": suction_body, "obj_qpos_adr": obj_qpos_adr,
            "rel_pos": rel_pos, "rel_quat": rel_quat}
    ignore_pairs = [(suction_body, obj_body)]
    goal_checker = planner.CollisionChecker(model, qpos_idx, held=held, ignore_body_pairs=ignore_pairs)

    if goal_checker.in_collision(q_target):
        # The pose ITSELF is reachable (IK just converged to it above) --
        # but this particular IK branch happens to sweep the arm through an
        # obstacle to get there. Damped-least-squares IK is a local solver:
        # from a given seed it converges to whichever nearby branch it falls
        # into, with no notion of collision, and different branches of the
        # same 6-DOF target pose (shoulder up/down, elbow flip, wrist flip)
        # can have very different swept volumes. A collision-free branch may
        # still exist elsewhere in joint space, so search for one instead of
        # failing immediately on the first (colliding) branch found.
        q_alt, conv2, ep2, er2 = _search_collision_free_ik(model, ctx, place_hover, place_quat, goal_checker, q_cur)
        if not conv2:
            report["failure_reason"] = "planner_no_collision_free_goal"
            return report
        q_target = q_alt

    path = planner.plan_path(model, qpos_idx, q_cur, q_target, joint_lo, joint_hi,
                              held=held, ignore_body_pairs=ignore_pairs)
    if path is None:
        report["failure_reason"] = "planner_failed_transit"
        return report

    for wp in path[1:]:
        status = _move_to(model, data, ctx, wp, budget, speed_scale, log=log)
        if status != "ok":
            report["failure_reason"] = f"move_failed_transit ({status})"
            return report
    report["phases"].append("transit")
    report["transit_waypoints"] = len(path)
    report["phases"].append("place_hover")
    q_cur = q_target

    # forbidden-contact check during transit
    forbidden = sc.get("forbidden_contacts", [])
    if forbidden:
        for fid in forbidden:
            fbody = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"fixture_{fid}")
            if _find_contact(model, data, obj_body, fbody) is not None:
                report["failure_reason"] = f"forbidden_contact_with_{fid}"
                return report

    # --- phase 5: descend to place ---
    q_target, conv, ep, er = ik.solve_ik_multistart(model, ctx.joint_names, ctx.suction_site,
                                                      place_pos, place_quat, q_init=q_cur, **IK_KWARGS)
    if not conv:
        report["failure_reason"] = f"ik_failed_place_contact (pos_err={ep:.4f}, rot_err={er:.1f}deg)"
        return report
    status = _move_to(model, data, ctx, q_target, budget, speed_scale * 0.3, log=log)
    if status != "ok":
        report["failure_reason"] = f"move_failed_place_descend ({status})"
        return report
    report["phases"].append("place_descend")
    q_cur = q_target

    # brief hold so contact settles before release
    if not _step_hold(model, data, ctrl_idx, data.ctrl[ctrl_idx].copy(), 100, budget, log):
        report["failure_reason"] = "timeout_before_release"
        return report

    # --- release suction ---
    data.eq_active[eq_id] = 0
    report["phases"].append("suction_released")

    # --- phase 6: retreat ---
    q_target, conv, ep, er = ik.solve_ik_multistart(model, ctx.joint_names, ctx.suction_site,
                                                      place_hover, place_quat, q_init=q_cur, **IK_KWARGS)
    if conv:
        _move_to(model, data, ctx, q_target, budget, speed_scale, log=log)
        report["phases"].append("retreat")

    # --- final settle: object must stay within tolerance & near-still for settle_time_s continuously ---
    pos_tol = sc["position_tolerance_m"]
    rot_tol = sc["orientation_tolerance_deg"]
    settle_needed = sc["settle_time_s"]
    settle_have = 0.0
    vel_thresh = 0.02  # m/s and rad/s, "near-zero" heuristic

    while budget.remaining > 0:
        if not _step_hold(model, data, ctrl_idx, data.ctrl[ctrl_idx].copy(), 1, budget, log):
            break
        obj_pos, obj_quat = _body_pose(data, obj_body)
        pos_err = np.linalg.norm(obj_pos - obj_target_pos)
        rot_err = quat_angle_deg(obj_quat, obj_target_quat)
        lin_vel = np.linalg.norm(data.cvel[obj_body][3:6])
        ang_vel = np.linalg.norm(data.cvel[obj_body][0:3])
        within = pos_err <= pos_tol and rot_err <= rot_tol and lin_vel < vel_thresh and ang_vel < vel_thresh
        if within:
            settle_have += model.opt.timestep
        else:
            settle_have = 0.0
        if settle_have >= settle_needed:
            report["success"] = True
            report["final_pos_error_m"] = float(pos_err)
            report["final_rot_error_deg"] = float(rot_err)
            report["phases"].append("settled")
            return report

    report["failure_reason"] = report["failure_reason"] or "timeout_waiting_for_settle"
    obj_pos, obj_quat = _body_pose(data, obj_body)
    report["final_pos_error_m"] = float(np.linalg.norm(obj_pos - obj_target_pos))
    report["final_rot_error_deg"] = float(quat_angle_deg(obj_quat, obj_target_quat))
    return report
