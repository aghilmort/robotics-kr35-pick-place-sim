#!/usr/bin/env python3
"""
Generates an MJCF model of a KUKA KR35 R1840-3 (CyberTech family) with a
suction end effector, from the real kinematic chain AND real meshes/inertial
data published by KUKA in their ROS2 description package:

  https://github.com/kroshu/kuka_robot_descriptions
  kuka_cybertech_support/urdf/kr35_r1840_3_hw_macro.xacro
  kuka_cybertech_support/meshes/kr35_r1840_3_hw/{visual,collision}/*

Joint origins, axes, limits, link masses, and link inertia tensors below are
copied directly from that xacro (not guessed). Link VISUAL and COLLISION
geometry are the real per-link meshes: visual meshes were converted from the
source .dae (Collada -- MuJoCo's native loader doesn't read Collada) to .stl
with `trimesh`, geometry unchanged; collision meshes are the source .stl
files, copied unmodified. See robot/meshes/kr35/ATTRIBUTION.md for license
and conversion details. MuJoCo convex-hulls each collision mesh automatically
at compile time (no manual convex decomposition needed).

This project previously modeled a KUKA KR16 R2010-2 with capsule/sphere
primitives standing in for missing real meshes (see git history /
claude/kr16_visual_fidelity_notes.md for that approach and the reasoning
behind it). KR35 R1840-3 replaces it here: same general CyberTech-family
kinematic layout, but a much larger 35kg-payload-class machine (vs KR16's
16kg) with real mesh + inertial data available, so this generator is a
different approach, not a port of the old one.

Convention note: URDF <origin rpy="r p y"> is a fixed-axis (extrinsic) X-Y-Z
rotation, i.e. R = Rz(yaw) @ Ry(pitch) @ Rx(roll). We reproduce that with
scipy's extrinsic-XYZ Euler convention so every body/geom/inertial pos+quat
in the MJCF tree matches the source xacro exactly.
"""
import math
from dataclasses import dataclass

from scipy.spatial.transform import Rotation


@dataclass
class JointSpec:
    name: str
    pos: tuple      # xyz offset of this joint's child body, in parent body frame (m)
    rpy: tuple      # fixed-axis (extrinsic) X-Y-Z rotation of the child body frame (rad)
    lower: float    # rad
    upper: float    # rad
    effort: float   # N*m
    velocity: float  # rad/s


@dataclass
class LinkInertial:
    mass: float
    com_pos: tuple   # xyz of the inertial frame origin, in this link's own body frame (m)
    com_rpy: tuple   # orientation of the inertial (principal-axis) frame (rad, fixed-axis XYZ)
    diag: tuple      # (ixx, iyy, izz) about the inertial frame above -- already diagonal
                      # in that frame per the source xacro (all off-diagonal terms are 0),
                      # which is exactly what MuJoCo's <inertial pos quat diaginertia> wants.


PI = math.pi

# Copied verbatim from kr35_r1840_3_hw_macro.xacro, joints 1-6.
JOINTS = [
    JointSpec("joint_1", (0.0, 0.0, 0.2638), (PI, 0.0, 0.0),
              -3.2288591161895095, 3.2288591161895095, 1920.0, PI),
    JointSpec("joint_2", (0.15, 0.0935, -0.1797), (PI / 2.0, 0.0, 0.0),
              -3.2288591161895095, 1.1344640137963142, 1721.0002677925002, PI),
    JointSpec("joint_3", (0.81, 0.0, -0.0225), (0.0, 0.0, 0.0),
              -2.0943951023931953, 3.1066860685499065, 692.79380081545, 3.490658503988659),
    JointSpec("joint_4", (0.3, -0.2, 0.116), (PI / 2.0, 0.0, -PI / 2.0),
              -PI, PI, 134.25883505388012, 6.981317007977318),
    JointSpec("joint_5", (0.0, -0.053, -0.5569), (0.0, PI / 2.0, PI / 2.0),
              -PI, PI, 152.7211182188344, 6.806784082777885),
    JointSpec("joint_6", (0.0515, 0.0, 0.053), (PI / 2.0, 0.0, -PI / 2.0),
              -6.2482787221397, 6.2482787221397, 68.63937567127184, 7.3303828583761845),
]

# Real per-link mass/inertia from the same xacro. base_link's inertial is
# applied to the base_link body below; link_1..6 map 1:1 onto JOINTS.
LINK_INERTIALS = {
    "base_link": LinkInertial(53.7647, (-0.008719211676062546, 6.228008340044944e-05, 0.15051911477233204),
                               (-0.11961611134062222, -0.42159909518503985, 0.010576198109219125),
                               (0.826710070988791, 1.0532984395003224, 1.067619390604537)),
    "link_1": LinkInertial(81.4493, (0.071941040196785, -0.01648269643974841, -0.10450396463566905),
                            (0.8528954774650944, 0.02231850440941273, -0.10893283726922436),
                            (1.6293306035988533, 1.48790696189089, 1.8648288850418433)),
    "link_2": LinkInertial(53.727, (0.3033584226738884, -0.0004796883689764923, -0.07022588261023321),
                            (-2.939567654685964, -1.4413963147644149, 1.372619729174395),
                            (5.991676420190299, 0.34656752938473534, 6.013413815236609)),
    "link_3": LinkInertial(33.3026, (0.11910403385921821, -0.0937917730147196, 0.09124612120374985),
                            (0.22136112456657464, -0.14964841802997608, 1.1505799825250325),
                            (0.5924680810605814, 0.46402433150527694, 0.7750079357706027)),
    "link_4": LinkInertial(15.3552, (-0.0001406340523080129, -0.07527848872043343, -0.2901270623632385),
                            (-3.100662238962031, -1.37804550206481, -1.5601621473383993),
                            (0.0553573080051956, 0.3143103119802837, 0.3247037705472359)),
    "link_5": LinkInertial(4.59, (0.00677, -5e-05, -0.00714),
                            (-0.012370262708191977, -0.8044745495411716, 3.137481150262563),
                            (0.008465962641296088, 0.011114663598543477, 0.012637373760160442)),
    "link_6": LinkInertial(1.24, (-1e-05, -1e-05, -0.0197),
                            (0.00017352122784772007, 0.00014634993607144222, -1.5662406035670715),
                            (0.0013290018473702423, 0.0012389981009168214, 0.002280000051712937)),
}

# Real per-link mesh pose from the same xacro's <visual>/<collision><origin>
# (identical for both -- the source describes one geometry, used for both
# roles). NOT zero, unlike the KR16 mesh set this project used before: these
# meshes are authored in a shared CAD-assembly reference frame, and this
# per-link origin is what re-composes them correctly under forward
# kinematics -- copied verbatim, not derived. See ATTRIBUTION.md.
MESH_ORIGINS = {
    "base_link": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "link_1": ((0.0, -3.230618256150718e-17, 0.26380000000000003), (-PI, 0.0, -0.0)),
    "link_2": ((-0.4435, -0.14999999999999997, 0.09350000000000006), (PI / 2.0, -0.0, PI / 2.0)),
    "link_3": ((-0.15000000000000005, 1.2535, 0.11600000000000006), (PI / 2.0, -0.0, -0.0)),
    "link_4": ((-1.4535, 3.824168447746776e-17, 0.44999999999999996), (1.8369701987210297e-16, PI / 2.0, 0.0)),
    "link_5": ((-1.0069000000000001, 1.4535, 0.053), (PI / 2.0, -3.749399456654644e-33, -6.123233995736766e-17)),
    "link_6": ((-1.4535000000000002, -6.025770166499968e-17, 1.0583999999999998), (1.2246467991473532e-16, PI / 2.0, 0.0)),
}

# ROS-Industrial 'flange' frame, fixed offset from link_6 (from the xacro's
# link_6-flange fixed joint). Much shorter than the KR16 build's stand-in
# stub (0.153m) because the real link_6 mesh already models the flange
# geometry itself -- this offset only reaches past the mesh's own tail.
FLANGE_POS = (0.0, 0.0, -0.0385)
FLANGE_RPY = (0.0, PI / 2.0, 0.0)

# suction_site's pos within the suction_cup body (identity orientation).
# The runner (trajectory.py) needs this to convert between the cup BODY's
# pose (what the suction weld equality constraint actually references) and
# the site's pose (what the IK solver targets) -- keep this in sync with
# the <site name="suction_site" .../> line below if it ever moves.
SUCTION_SITE_LOCAL_POS = (0.0, 0.0, 0.008)

# KUKA house orange, used to paint every real link mesh uniformly. The
# source .dae files carry their own per-submesh materials/textures, but
# those were not preserved through the dae->stl conversion (see
# ATTRIBUTION.md) -- painting everything one color is a deliberate,
# documented simplification, not a bug.
KUKA_ORANGE = "0.96 0.42 0.02 1"

LINK_NAMES = ["base_link"] + [f"link_{i+1}" for i in range(len(JOINTS))]


def rpy_to_quat_wxyz(rpy):
    """URDF fixed-axis (extrinsic) X-Y-Z rpy -> MuJoCo quaternion (w x y z).

    scipy's convention is the opposite of what it looks like at a glance:
    lowercase axis letters ('xyz') are EXTRINSIC (fixed-axis) rotations,
    uppercase ('XYZ') are INTRINSIC. Verified numerically against the
    explicit R = Rz(yaw) @ Ry(pitch) @ Rx(roll) URDF definition -- 'xyz'
    matches, 'XYZ' does not. (The KR16 build this project used earlier had
    this backwards and it was never caught, because every rpy value in that
    xacro happened to be single-axis, where intrinsic/extrinsic are
    identical -- KR35's joint_4/joint_6 and several MESH_ORIGINS entries are
    genuinely multi-axis and immediately exposed it as scattered, detached
    link meshes on the first render.)
    """
    r, p, y = rpy
    rot = Rotation.from_euler("xyz", [r, p, y])  # lowercase = extrinsic in scipy
    x, y_, z, w = rot.as_quat()  # scipy returns (x, y, z, w)
    return (w, x, y_, z)


def fmt(vec):
    return " ".join(f"{v:.6f}" for v in vec)


def build_mjcf(with_suction: bool = True) -> str:
    lines = []
    lines.append('<mujoco model="kr35_r1840_3_hw">')
    lines.append('  <compiler angle="radian" autolimits="true"/>')
    lines.append('  <option gravity="0 0 -9.81" integrator="implicitfast"/>')

    lines.append('  <asset>')
    for name in LINK_NAMES:
        lines.append(f'    <mesh name="{name}_visual" file="meshes/kr35/visual/{name}.stl"/>')
        lines.append(f'    <mesh name="{name}_collision" file="meshes/kr35/collision/{name}.stl"/>')
    lines.append('  </asset>')

    lines.append('  <default>')
    lines.append('    <default class="kr35_visual">')
    # Real mesh, rendered only -- no physics role. Collision is handled
    # separately by the (much cheaper, convex-hulled) collision mesh below.
    lines.append(f'      <geom type="mesh" rgba="{KUKA_ORANGE}" contype="0" conaffinity="0" group="2"/>')
    lines.append('    </default>')
    lines.append('    <default class="kr35_collision">')
    # Real collision mesh -- MuJoCo convex-hulls this at compile time. group
    # 3 keeps it out of normal renders (matching the KR16 build's
    # kr16_collision convention) while still being the actual physics geom.
    lines.append('      <geom type="mesh" group="3" rgba="1 0 0 0.3"/>')
    lines.append('    </default>')
    lines.append('  </default>')

    lines.append('  <worldbody>')
    indent = '    '
    # Unlike the KR16 build (which had no base mesh and invented a
    # BASE_PEDESTAL_HEIGHT capsule stand-in), the real base_link mesh here
    # already spans from ~floor level (bbox z-min ~= -0.003m, confirmed by
    # inspection) up to the shoulder -- no elevation needed, base_link sits
    # directly at the world origin like the source xacro's own base_link ->
    # base fixed joint (identity transform) implies.
    lines.append(f'{indent}<body name="base_link" pos="0 0 0">')
    li = LINK_INERTIALS["base_link"]
    iw, ix, iy, iz = rpy_to_quat_wxyz(li.com_rpy)
    lines.append(f'{indent}  <inertial pos="{fmt(li.com_pos)}" quat="{iw:.6f} {ix:.6f} {iy:.6f} {iz:.6f}" '
                  f'mass="{li.mass:.4f}" diaginertia="{fmt(li.diag)}"/>')
    mp, mr = MESH_ORIGINS["base_link"]
    mw, mx, my, mz = rpy_to_quat_wxyz(mr)
    lines.append(f'{indent}  <geom class="kr35_visual" mesh="base_link_visual" pos="{fmt(mp)}" '
                  f'quat="{mw:.6f} {mx:.6f} {my:.6f} {mz:.6f}"/>')
    lines.append(f'{indent}  <geom class="kr35_collision" mesh="base_link_collision" pos="{fmt(mp)}" '
                  f'quat="{mw:.6f} {mx:.6f} {my:.6f} {mz:.6f}"/>')

    # Recursively nest joint_1..joint_6 bodies, each carrying its own real
    # mesh pair, matching the URDF chain exactly.
    body_indent = indent + '  '
    for i, j in enumerate(JOINTS):
        link_name = f"link_{i+1}"
        w, x, y, z = rpy_to_quat_wxyz(j.rpy)
        lines.append(f'{body_indent}<body name="{link_name}" pos="{fmt(j.pos)}" quat="{w:.6f} {x:.6f} {y:.6f} {z:.6f}">')
        lines.append(f'{body_indent}  <joint name="{j.name}" type="hinge" axis="0 0 1" '
                      f'range="{j.lower:.6f} {j.upper:.6f}" damping="40.0" frictionloss="4.0" armature="0.15"/>')
        li = LINK_INERTIALS[link_name]
        iw, ix, iy, iz = rpy_to_quat_wxyz(li.com_rpy)
        lines.append(f'{body_indent}  <inertial pos="{fmt(li.com_pos)}" quat="{iw:.6f} {ix:.6f} {iy:.6f} {iz:.6f}" '
                      f'mass="{li.mass:.4f}" diaginertia="{fmt(li.diag)}"/>')
        mp, mr = MESH_ORIGINS[link_name]
        mw, mx, my, mz = rpy_to_quat_wxyz(mr)
        lines.append(f'{body_indent}  <geom class="kr35_visual" mesh="{link_name}_visual" pos="{fmt(mp)}" '
                      f'quat="{mw:.6f} {mx:.6f} {my:.6f} {mz:.6f}"/>')
        lines.append(f'{body_indent}  <geom class="kr35_collision" mesh="{link_name}_collision" pos="{fmt(mp)}" '
                      f'quat="{mw:.6f} {mx:.6f} {my:.6f} {mz:.6f}"/>')
        body_indent += '  '

    # Flange + suction gripper, attached to link_6. The gripper itself has
    # no real KUKA mesh (it's not part of the arm) -- kept as the same
    # simple primitive stack the KR16 build used, just relocated onto
    # KR35's real (much shorter) flange offset.
    fw, fx, fy, fz = rpy_to_quat_wxyz(FLANGE_RPY)
    lines.append(f'{body_indent}<body name="flange" pos="{fmt(FLANGE_POS)}" quat="{fw:.6f} {fx:.6f} {fy:.6f} {fz:.6f}">')
    lines.append(f'{body_indent}  <site name="flange_site" size="0.01" rgba="0 1 0 1"/>')
    lines.append(f'{body_indent}  <geom type="cylinder" size="0.06 0.014" pos="0 0 0" '
                  f'rgba="0.08 0.08 0.09 1" contype="0" conaffinity="0"/>')
    if with_suction:
        lines.append(f'{body_indent}  <body name="suction_gripper" pos="0 0 0">')
        lines.append(f'{body_indent}    <geom type="cylinder" size="0.035 0.06" pos="0 0 0.06" '
                      f'rgba="0.15 0.15 0.15 1" contype="0" conaffinity="0"/>')
        lines.append(f'{body_indent}    <body name="suction_cup" pos="0 0 0.15">')
        lines.append(f'{body_indent}      <geom name="suction_cup_geom" type="cylinder" size="0.025 0.008" '
                      f'rgba="0.05 0.05 0.05 1" friction="1.0 0.01 0.001" solimp="0.95 0.99 0.001" solref="0.004 1"/>')
        lines.append(f'{body_indent}      <site name="suction_site" pos="0 0 0.008" size="0.005" rgba="1 0 0 1"/>')
        lines.append(f'{body_indent}    </body>')
        lines.append(f'{body_indent}  </body>')
    lines.append(f'{body_indent}</body>')

    for i in range(len(JOINTS)):
        body_indent = body_indent[:-2]
        lines.append(f'{body_indent}</body>')
    lines.append(f'{indent}</body>')  # base_link
    lines.append('  </worldbody>')

    # Exclude all pairwise self-collision among the arm's own link bodies.
    # MuJoCo already skips collision between directly-connected (parent/
    # child) body pairs by default; this covers the non-adjacent pairs too.
    # The real meshes are far more accurate than the old capsules, but a
    # folded-up pose can still bring non-adjacent real link meshes into
    # contact in ways that aren't a meaningful self-collision to model here.
    lines.append('  <contact>')
    for a in range(len(LINK_NAMES)):
        for b in range(a + 1, len(LINK_NAMES)):
            lines.append(f'    <exclude body1="{LINK_NAMES[a]}" body2="{LINK_NAMES[b]}"/>')
    lines.append('  </contact>')

    # Position-actuator gains, empirically tuned (not guessed) against
    # KR35's real link masses (243kg total vs. the KR16 build's placeholder
    # ~90kg) -- see robot/tune_gains.py for the settle-test this was
    # validated against, and claude/design-notes.md for the full story.
    # forcerange caps each actuator at that JOINT's real KUKA effort limit
    # (JOINTS[i].effort) -- the KR16 build never set this (position
    # actuators default to unlimited force), which was harmless there, but
    # was a real bug waiting to happen here: at a naive first-pass kp of
    # 400000, an unbounded actuator could slam out enormous corrective
    # torque from a one-timestep numerical bump (e.g. the suction weld
    # deactivating while still in contact) -- observed directly as the
    # object being explosively flung the instant suction released. Real
    # KUKA joints cannot exceed their rated effort either way, so capping
    # it is a correctness fix, not just a stability band-aid.
    #
    # Capping torque exposed a second, subtler issue: at kp=400000 (tuned
    # only against 3 static gravity-hold poses, never against a full
    # episode) some reachable configurations -- particularly ones the
    # RRT-connect transit planner's joint-space paths land on, not anything
    # in the simpler unobstructed basic_pick_place.yaml -- drove every one
    # of the 6 actuators into simultaneous torque saturation, oscillating
    # between +limit and -limit (an underdamped instability, not a genuine
    # "not enough real torque" case) and stalling the arm well short of its
    # commanded pose. Before the forcerange fix this same instability just
    # meant unbounded torque, i.e. the same flinging bug wearing a
    # different hat. kp=200000/kv=12000 (vs. the naive 400000/20000) was
    # chosen by sweeping candidate gains against obstacle_pick_place.yaml
    # batches (not just the static settle test) and picking the most robust
    # -- it still holds all three settle-test poses to well under a degree,
    # see tune_gains.py.
    lines.append('  <actuator>')
    for j in JOINTS:
        lines.append(f'    <position name="act_{j.name}" joint="{j.name}" '
                      f'kp="200000" kv="12000" ctrlrange="{j.lower:.6f} {j.upper:.6f}" '
                      f'forcerange="{-j.effort:.3f} {j.effort:.3f}"/>')
    lines.append('  </actuator>')

    lines.append('  <sensor>')
    if with_suction:
        lines.append('    <touch name="suction_touch" site="suction_site"/>')
    for j in JOINTS:
        lines.append(f'    <jointpos name="sensor_{j.name}_pos" joint="{j.name}"/>')
    lines.append('  </sensor>')

    lines.append('</mujoco>')
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import pathlib
    xml = build_mjcf()
    out = pathlib.Path(__file__).parent / "kr35_r1840_3_hw.xml"
    out.write_text(xml)
    print(f"wrote {out} ({len(xml)} bytes)")
