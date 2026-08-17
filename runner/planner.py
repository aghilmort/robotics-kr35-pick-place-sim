"""Joint-space RRT-Connect motion planner with real MuJoCo collision
checking.

Used for exactly one thing: the "transit" leg of an episode (arm carrying
the picked object from above the pick point to above the place point),
which is the only part of a scripted pick-and-place episode where an
obstacle sitting *between* the two task locations can actually matter. The
short waypoint-to-waypoint moves elsewhere in the episode (descending onto
/ lifting off a known surface, right next to a known target) still use
plain joint-space interpolation in trajectory.py -- there's no useful
planning problem in a 5cm straight-down move, and running collision-checked
search there would only add cost with no benefit.

Collision checking runs on a dedicated scratch MjData, created once per
plan() call and never touched by anything else -- it must never be the live
simulation `data` the rest of the episode is stepping. Each check uses
mj_kinematics + mj_collision only (position + contact detection), not a
full mj_forward/mj_step -- there's no dynamics to compute, so this is cheap
enough to call thousands of times per plan.

Self-collision between the arm's own links is already excluded at the
model-compile level (see the <exclude> tags build_kr35_mjcf.py adds --
scene_builder.py leaves those untouched when attaching the robot).
The one pair this module must additionally ignore by hand is the suction
cup against whatever object it is currently holding: that pair is expected
to be in (near-)contact for the entire transit, since the object is rigidly
welded to the cup, and treating it as an obstacle would make every carrying
motion un-plannable.
"""
import numpy as np
import mujoco

from geometry import compose


class CollisionChecker:
    """Wraps a scratch MjData + the bookkeeping needed to test one arm
    configuration (and, if an object is being carried, that object's
    resulting pose) for collisions against the rest of the scene.
    """

    def __init__(self, model, qpos_idx, held=None, ignore_body_pairs=()):
        """
        qpos_idx: qpos addresses of the arm's 6 joints (same ordering as the
          q vectors passed to plan_path).
        held: None, or a dict describing an object rigidly attached to the
          suction cup for the duration of planning --
            {"cup_body_id": int, "obj_qpos_adr": int,
             "rel_pos": (3,), "rel_quat": (4,)}
          rel_pos/rel_quat is the object's pose expressed in the cup body's
          frame (same convention as the live suction weld's eq_data, and
          the same value trajectory.py already computes at engage time).
        ignore_body_pairs: iterable of (body_id_a, body_id_b) pairs whose
          mutual contact should never count as a collision (e.g. cup vs.
          held object).
        """
        self.model = model
        self.data = mujoco.MjData(model)
        self.qpos_idx = np.asarray(qpos_idx)
        self.held = held
        self.ignore_pairs = {frozenset(p) for p in ignore_body_pairs}

    def in_collision(self, q):
        m, d = self.model, self.data
        d.qpos[self.qpos_idx] = q
        mujoco.mj_kinematics(m, d)
        if self.held is not None:
            cup_id = self.held["cup_body_id"]
            cup_pos = d.xpos[cup_id].copy()
            cup_quat = np.zeros(4)
            mujoco.mju_mat2Quat(cup_quat, d.xmat[cup_id])
            obj_pos, obj_quat = compose(cup_pos, cup_quat, self.held["rel_pos"], self.held["rel_quat"])
            adr = self.held["obj_qpos_adr"]
            d.qpos[adr:adr + 3] = obj_pos
            d.qpos[adr + 3:adr + 7] = obj_quat
            mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        for i in range(d.ncon):
            c = d.contact[i]
            ba = m.geom_bodyid[c.geom1]
            bb = m.geom_bodyid[c.geom2]
            if frozenset((ba, bb)) in self.ignore_pairs:
                continue
            return True
        return False

    def edge_free(self, q_from, q_to, resolution=0.05):
        """Collision-check the straight joint-space segment q_from->q_to,
        sampled every `resolution` radians (in the segment's own L2 length)
        -- not just the endpoint, so a thin obstacle can't be tunnelled
        through between two widely-spaced samples."""
        dist = float(np.linalg.norm(q_to - q_from))
        n = max(1, int(np.ceil(dist / resolution)))
        for i in range(1, n + 1):
            q = q_from + (q_to - q_from) * (i / n)
            if self.in_collision(q):
                return False
        return True


class _Tree:
    def __init__(self, root):
        self.configs = [np.asarray(root, dtype=float)]
        self.parents = [-1]

    def nearest(self, q):
        # Linear scan: trees here run to at most a few thousand nodes for a
        # 6-DOF arm, well within "fine without a kd-tree" territory for an
        # offline scripted runner.
        dists = [np.linalg.norm(q - c) for c in self.configs]
        i = int(np.argmin(dists))
        return i, self.configs[i]

    def add(self, q, parent_idx):
        self.configs.append(np.asarray(q, dtype=float))
        self.parents.append(parent_idx)
        return len(self.configs) - 1

    def path_to_root(self, idx):
        out = []
        while idx != -1:
            out.append(self.configs[idx])
            idx = self.parents[idx]
        return out  # leaf -> root order


REACHED, ADVANCED, TRAPPED = "reached", "advanced", "trapped"


def _extend(tree, q_target, checker, step_size, edge_resolution):
    idx, q_near = tree.nearest(q_target)
    delta = q_target - q_near
    dist = float(np.linalg.norm(delta))
    if dist < 1e-9:
        return REACHED, idx
    if dist <= step_size:
        q_new = q_target
    else:
        q_new = q_near + delta * (step_size / dist)
    if not checker.edge_free(q_near, q_new, edge_resolution):
        return TRAPPED, idx
    new_idx = tree.add(q_new, idx)
    return (REACHED if dist <= step_size else ADVANCED), new_idx


def _connect(tree, q_target, checker, step_size, edge_resolution):
    """Repeatedly extend `tree` toward q_target (not just one step) until it
    either reaches q_target exactly or gets blocked -- the "greedy" half of
    RRT-Connect that makes it converge much faster than plain RRT."""
    status, idx = _extend(tree, q_target, checker, step_size, edge_resolution)
    while status == ADVANCED:
        status, idx = _extend(tree, q_target, checker, step_size, edge_resolution)
    return status, idx


def _shortcut(path, checker, edge_resolution, iters, rng):
    """Randomized shortcutting: repeatedly try to splice out an interior
    run of the path by connecting two farther-apart points directly, if
    that direct segment is collision-free. Standard RRT post-processing --
    raw tree-growth paths are jagged (every step is toward a random
    sample), this trims that down to something closer to a direct route
    without giving up the collision guarantees (every replacement segment
    is itself re-checked)."""
    path = list(path)
    for _ in range(iters):
        if len(path) <= 2:
            break
        i, j = sorted(rng.integers(0, len(path), size=2))
        if j - i < 2:
            continue
        if checker.edge_free(path[i], path[j], edge_resolution):
            path = path[:i + 1] + path[j:]
    return path


def plan_path(model, qpos_idx, q_start, q_goal, joint_lo, joint_hi,
              held=None, ignore_body_pairs=(), max_iters=4000, step_size=0.25,
              edge_resolution=0.05, goal_bias=0.1, shortcut_iters=150, seed=0):
    """RRT-Connect in joint space. Returns a list of joint configs from
    q_start to q_goal (inclusive) if a collision-free path is found within
    max_iters, else None.

    q_start is assumed collision-free (it's wherever the arm already is,
    mid-episode). q_goal is checked explicitly up front so a bad IK
    solution fails fast with a clear reason rather than burning the whole
    iteration budget searching for something unreachable.
    """
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)
    joint_lo = np.asarray(joint_lo, dtype=float)
    joint_hi = np.asarray(joint_hi, dtype=float)
    rng = np.random.default_rng(seed)

    checker = CollisionChecker(model, qpos_idx, held=held, ignore_body_pairs=ignore_body_pairs)
    if checker.in_collision(q_goal):
        return None

    tree_a = _Tree(q_start)   # rooted at start
    tree_b = _Tree(q_goal)    # rooted at goal
    swapped = False           # tracks which of tree_a/tree_b is currently the "start" tree

    for _ in range(max_iters):
        q_rand = q_goal if rng.random() < goal_bias else rng.uniform(joint_lo, joint_hi)
        status_a, idx_a = _extend(tree_a, q_rand, checker, step_size, edge_resolution)
        if status_a != TRAPPED:
            status_b, idx_b = _connect(tree_b, tree_a.configs[idx_a], checker, step_size, edge_resolution)
            if status_b == REACHED:
                start_tree, start_idx = (tree_a, idx_a) if not swapped else (tree_b, idx_b)
                goal_tree, goal_idx = (tree_b, idx_b) if not swapped else (tree_a, idx_a)
                head = list(reversed(start_tree.path_to_root(start_idx)))   # start -> meet
                tail = goal_tree.path_to_root(goal_idx)[1:]                  # meet(excl) -> goal... wait see note
                # goal_tree is rooted at q_goal, so path_to_root(goal_idx) goes
                # meet-node -> ... -> q_goal (leaf-to-root order, root=q_goal).
                # That's already meet->goal order, no reversal needed; drop
                # index 0 (the meeting config itself) to avoid a duplicate.
                path = head + tail
                return _shortcut(path, checker, edge_resolution, shortcut_iters, rng)
        tree_a, tree_b = tree_b, tree_a
        swapped = not swapped

    return None
