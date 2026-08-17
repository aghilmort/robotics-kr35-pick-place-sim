"""Assembles a scenario YAML dict into a compiled MuJoCo model.

Uses mujoco.MjSpec to compose three pieces at the spec (pre-compile) level:
  1. a "world" spec: ground plane + static fixtures + dynamic objects
  2. the robot spec, loaded from robot/kr35_r1840_3_hw.xml and attached at a
     frame placed at scenario.robot.base_pose (bodies/joints/sites get a
     "robot_" prefix after attach)
  3. one inactive weld equality constraint per graspable object, connecting
     the (prefixed) suction cup body to that object's body. The runner
     activates/deactivates these at runtime and rewrites their relpose to
     the live relative transform captured at the moment of contact -- see
     runner.py / suction.py.

Returns the compiled model/data plus a SceneContext with all the name/id
bookkeeping the trajectory planner and success checker need.
"""
import pathlib
from dataclasses import dataclass, field

import mujoco
import numpy as np

from geometry import euler_deg_to_quat, quat_mul

ROBOT_XML = {
    "kr35_r1840_3_hw": pathlib.Path(__file__).parent.parent / "robot" / "kr35_r1840_3_hw.xml",
}

SHAPE_TO_GEOM = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
}

ROBOT_PREFIX = "robot_"
N_ARM_JOINTS = 6


@dataclass
class ObjectInfo:
    id: str
    body_name: str
    shape: str
    size: np.ndarray
    nominal_pos: np.ndarray
    nominal_quat: np.ndarray
    graspable: bool
    eq_id: int | None = None


@dataclass
class SceneContext:
    scenario: dict
    objects: dict[str, ObjectInfo] = field(default_factory=dict)
    fixture_top: dict[str, tuple[np.ndarray, float]] = field(default_factory=dict)  # id -> (world top-center pos, yaw rad)
    joint_names: list = field(default_factory=list)
    suction_cup_body: str = ROBOT_PREFIX + "suction_cup"
    suction_site: str = ROBOT_PREFIX + "suction_site"
    flange_site: str = ROBOT_PREFIX + "flange_site"
    rng: np.random.Generator = None


def _top_offset_local(shape: str, size):
    if shape == "box":
        return np.array([0.0, 0.0, size[2]])
    if shape == "cylinder":
        return np.array([0.0, 0.0, size[1]])
    if shape == "sphere":
        return np.array([0.0, 0.0, size[0]])
    raise ValueError(f"unsupported shape {shape}")


def build_model(scenario: dict):
    ctx = SceneContext(scenario=scenario)
    seed = scenario.get("seed")
    ctx.rng = np.random.default_rng(seed)

    world = mujoco.MjSpec()
    world.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    world.option.gravity = [0, 0, -9.81]

    gf = scenario["scene"].get("ground_friction", [1.0, 0.005, 0.0001])
    world.worldbody.add_geom(
        name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[5, 5, 0.1], friction=gf, rgba=[0.5, 0.5, 0.55, 1.0],
    )

    # --- static fixtures ---
    for fx in scenario["scene"].get("fixtures", []):
        pos = np.array(fx["pose"].get("pos", [0, 0, 0]), dtype=float)
        rpy_deg = fx["pose"].get("rpy_deg", [0, 0, 0])
        quat = euler_deg_to_quat(rpy_deg)
        size = np.array(fx["size"], dtype=float)
        body = world.worldbody.add_body(name=f"fixture_{fx['id']}", pos=pos, quat=quat)
        body.add_geom(
            type=SHAPE_TO_GEOM[fx["shape"]], size=_pad_size(fx["shape"], size),
            rgba=fx.get("color_rgba", [0.6, 0.6, 0.6, 1.0]),
        )
        top_local = _top_offset_local(fx["shape"], size)
        top_world = pos + _rotate_z(top_local, np.radians(rpy_deg[2]))
        ctx.fixture_top[fx["id"]] = (top_world, np.radians(rpy_deg[2]))

    # --- dynamic objects ---
    for obj in scenario["scene"]["objects"]:
        size = np.array(obj["size"], dtype=float)
        nominal_pos = np.array(obj["pose"].get("pos", [0, 0, 0]), dtype=float)
        nominal_rpy = obj["pose"].get("rpy_deg", [0, 0, 0])
        nominal_quat = euler_deg_to_quat(nominal_rpy)

        randomize = obj.get("randomize")
        if randomize:
            pos_range = randomize.get("pos_range")
            if pos_range:
                nominal_pos = np.array([ctx.rng.uniform(lo, hi) for lo, hi in pos_range])
            yaw_range = randomize.get("yaw_range_deg")
            if yaw_range:
                yaw = ctx.rng.uniform(yaw_range[0], yaw_range[1])
                nominal_quat = euler_deg_to_quat([0, 0, yaw])

        body_name = f"object_{obj['id']}"
        body = world.worldbody.add_body(name=body_name, pos=nominal_pos, quat=nominal_quat)
        body.add_freejoint()
        geom = body.add_geom(
            type=SHAPE_TO_GEOM[obj["shape"]], size=_pad_size(obj["shape"], size),
            rgba=obj.get("color_rgba", [0.7, 0.7, 0.7, 1.0]),
        )
        geom.mass = obj["mass_kg"]
        if "friction" in obj:
            geom.friction = obj["friction"]

        ctx.objects[obj["id"]] = ObjectInfo(
            id=obj["id"], body_name=body_name, shape=obj["shape"], size=size,
            nominal_pos=nominal_pos, nominal_quat=nominal_quat,
            graspable=obj.get("graspable", True),
        )

    # --- robot, attached at base_pose ---
    robot_cfg = scenario["robot"]
    robot_xml = ROBOT_XML[robot_cfg["model"]]
    base_pose = robot_cfg.get("base_pose", {})
    base_pos = np.array(base_pose.get("pos", [0, 0, 0]), dtype=float)
    base_quat = euler_deg_to_quat(base_pose.get("rpy_deg", [0, 0, 0]))
    mount = world.worldbody.add_frame(pos=base_pos, quat=base_quat)

    robot_spec = mujoco.MjSpec.from_file(str(robot_xml))
    world.attach(robot_spec, frame=mount, prefix=ROBOT_PREFIX)
    ctx.joint_names = [f"{ROBOT_PREFIX}joint_{i+1}" for i in range(N_ARM_JOINTS)]

    # --- suction weld candidates, one per graspable object ---
    for obj_id, info in ctx.objects.items():
        if not info.graspable:
            continue
        eq = world.add_equality(
            type=mujoco.mjtEq.mjEQ_WELD,
            name=f"suction_weld_{obj_id}",
            name1=ctx.suction_cup_body,
            name2=info.body_name,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            active=False,
            data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # anchor=0, relpose=identity, torquescale=1
            solref=[0.004, 1.0],
        )
        info.eq_id = eq  # resolved to an integer id after compile, see below

    model = world.compile()
    data = mujoco.MjData(model)

    # resolve eq ids by name now that the model is compiled
    for obj_id, info in ctx.objects.items():
        if info.graspable:
            info.eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"suction_weld_{obj_id}")

    # place objects (qpos overrides the spec's nominal pos, needed because
    # freejoint qpos is what actually drives simulation state)
    mujoco.mj_resetData(model, data)
    for obj_id, info in ctx.objects.items():
        adr = _freejoint_qpos_adr(model, info.body_name)
        data.qpos[adr:adr + 3] = info.nominal_pos
        data.qpos[adr + 3:adr + 7] = info.nominal_quat

    # robot home pose
    home = robot_cfg.get("home_joint_positions", [0.0] * N_ARM_JOINTS)
    for name, val in zip(ctx.joint_names, home):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = val
        data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{ROBOT_PREFIX}act_{name[len(ROBOT_PREFIX):]}")] = val

    mujoco.mj_forward(model, data)
    return model, data, ctx


def _freejoint_qpos_adr(model, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    joint_id = model.body_jntadr[body_id]
    assert model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE, (
        f"expected {body_name} to have a free joint"
    )
    return model.jnt_qposadr[joint_id]


def _pad_size(shape, size):
    # MuJoCo requires fixed-length size arrays per geom type even though our
    # schema allows shorter arrays (sphere=1, cylinder=2, box=3).
    if shape == "sphere":
        return [size[0], 0, 0]
    if shape == "cylinder":
        return [size[0], size[1], 0]
    return list(size)


def _rotate_z(vec, yaw_rad):
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    x, y, z = vec
    return np.array([c * x - s * y, s * x + c * y, z])
