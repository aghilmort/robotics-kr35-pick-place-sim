# Pick-and-Place Scenario Runner (KUKA KR35 R1840-3 + suction)

A local scenario schema + MuJoCo runner for scripted pick-and-place
episodes, built around a KUKA KR CYBERTECH (KR35 R1840-3) arm with a
suction end effector, using KUKA's real published meshes for both
rendering and collision.

## Layout

```
robot/
  build_kr35_mjcf.py    generates the robot MJCF from real KUKA joint/mesh/inertial data
  kr35_r1840_3_hw.xml    generated output (run build_kr35_mjcf.py to refresh)
  tune_gains.py          empirical actuator-gain settle test (see "Design overview" below)
  meshes/kr35/            real KUKA visual + collision meshes (see ATTRIBUTION.md)
schema/
  scenario.schema.json  JSON Schema for scenario YAML files
  examples/
    basic_pick_place.yaml
    obstacle_pick_place.yaml   same task, with a wall between pick and place
runner/
  geometry.py           quaternion/pose helpers
  scene_builder.py       scenario YAML -> compiled MuJoCo model (mujoco.MjSpec)
  ik.py                  damped-least-squares IK w/ random-restart fallback
  planner.py              joint-space RRT-Connect w/ real collision checking (transit leg only)
  trajectory.py           waypoint state machine, suction engage/release, success eval
  runner.py               CLI: validate, run, batch, optional MP4 video
```

## Running it

```
cd runner
python3 runner.py ../schema/examples/basic_pick_place.yaml
python3 runner.py ../schema/examples/basic_pick_place.yaml --trials 30 --seed-from 0
python3 runner.py ../schema/examples/basic_pick_place.yaml --video out.mp4
```

Exit code is 0 on success (or all trials succeeding in batch mode), 1 otherwise.
A batch run prints a summary with `success_rate` and a `failure_reasons` histogram.

## Demo

`obstacle_pick_place.yaml`: the transit-leg RRT-Connect planner routing the
arm sideways around a wall standing between the table and the tray, while
holding the picked cube.

<video src="renders/kr35_obstacle_demo.mp4" controls width="480">
  Your viewer can't play inline video -- see <code>renders/kr35_obstacle_demo.mp4</code> directly.
</video>

![KR35 arm mid-transit, holding the picked cube, routing around the wall obstacle toward the tray](renders/kr35_obstacle_demo_frame.png)

## Design overview

- **Robot**: KUKA KR35 R1840-3 (CyberTech family, 35kg payload class),
  modeled from KUKA's real published xacro/mesh data
  (`kroshu/kuka_robot_descriptions`, Apache-2.0) -- real joint kinematics,
  per-link mass/inertia, and real visual + collision meshes, not
  approximated primitives. See `robot/build_kr35_mjcf.py` and
  `robot/meshes/kr35/ATTRIBUTION.md`.
- **Suction**: MuJoCo has no vacuum physics, so grasping is a
  contact-triggered `weld` equality constraint -- engages on contact,
  releases on command. See `trajectory.py` around `suction_engaged` /
  `suction_released`.
- **Schema scope**: one YAML file = one full runnable episode (scene +
  robot config + task + success criteria), not just a bare goal spec. See
  `schema/scenario.schema.json`.
- **Motion**: scripted, not a learned/pluggable policy -- IK per waypoint,
  driven via smoothstep joint-space interpolation timed from the KR35's
  real per-joint velocity limits.
- **Transit planning**: the pick_lift -> place_hover crossing goes through
  a real joint-space RRT-Connect planner (`runner/planner.py`) with genuine
  MuJoCo collision checking, rather than a hand-picked "fly higher"
  waypoint -- see `schema/examples/obstacle_pick_place.yaml` and the demo
  above.

The debugging history behind these choices -- the `.dae`-to-`.stl` mesh
conversion, a scipy euler-convention bug, and the actuator torque-limit
tuning story referenced in "Known limitations" below -- is kept in the
team's internal working notes rather than published here.

## Known limitations (intentional, for a first pass)

- **The obstacle scenario (`obstacle_pick_place.yaml`) succeeds on roughly
  ~73% of trials (44/60 measured), noticeably lower than
  `basic_pick_place.yaml`'s 100% (40/40).** All observed failures are the
  same story: the RRT-planner's transit path lands the arm in a
  configuration where reaching the exact commanded place pose would
  require more instantaneous torque than the (now correctly capped, see
  above) real KUKA effort limits allow, the arm stalls a few centimeters
  short, and the object is released from that stalled position and
  tumbles on landing -- a real, torque-limited-dynamics failure mode that
  structurally could not happen in the old KR16 build (which had no
  torque limit at all). This is arguably *more* physically honest than the
  old model, not a regression in the underlying physics -- but it is a
  real accuracy gap worth closing. The gain (kp=200000/kv=12000) was
  chosen empirically as the most robust of several candidates tried
  against full episode batches, not just the static settle test; a proper
  fix would need the motion executor (`_move_to` in `trajectory.py`) to
  respect available torque when timing a trajectory, not just each
  joint's rated velocity limit -- see "Suggested next steps".
- **Arm/environment collision is now driven by KUKA's real (convex-hulled)
  collision meshes**, a meaningful accuracy upgrade over the old capsule
  approximation -- but still convex per-link (no concave decomposition),
  and self-collision between the arm's own links is excluded entirely (see
  `build_kr35_mjcf.py`'s `<exclude>` tags), the same simplification the
  capsule model made.
- **The transit leg is obstacle-aware (RRT-Connect + real collision
  checking); the short pick/place descend/lift moves are not.** That's a
  deliberate scope choice, not an oversight -- but it does mean an
  obstacle placed very close to the pick or place point itself, inside the
  short direct-interpolation segments, still isn't handled.
- **The transit planner's goal selection is a bounded random-restart
  search, not an exhaustive one** (see `_search_collision_free_ik` in
  `trajectory.py` and `MAX_GOAL_JUMP`). For a 6-DOF arm the collision-free
  IK branch for a given target pose can sit in a narrow enough slice of
  joint space that even ~60 random restarts occasionally miss it. A proper
  fix would need analytic (closed-form) IK for this arm to enumerate its
  actual finite branch set directly, instead of hoping random restarts
  land near a good one.
- **Straight-line-ish, not Cartesian, motion.** Joint-space smoothstep
  interpolation between IK solutions (or, for the transit leg, between
  planner waypoints), not a Cartesian-space path -- fine for open
  workspace, not verified for cluttered scenes.
- Only box/cylinder/sphere primitive objects and fixtures; no mesh objects
  (the robot itself is now mesh-based, but scenario-authored fixtures and
  graspable objects are still primitives). Fixtures are a single solid
  box/cylinder (no separate legs/thin-top), so a "table" needs to be built
  as a thin slab at the right height, not a full block.
- Only straight-down-style single suction cup grasps; no multi-suction-cup
  or force/torque-limited grasp failure modeling (payload check is a mass
  threshold, not a real suction-seal physics check).

## Suggested next steps

- Make `_move_to`'s trajectory timing torque-aware (e.g. a trapezoidal /
  S-curve velocity profile budgeted against each joint's real effort
  limit and the arm's actual inertia at that pose), rather than timing
  purely from rated joint velocity -- the direct fix for the obstacle
  scenario's ~27% failure rate described above.
- Add mesh-based collision for scenario-authored objects/fixtures (the
  robot itself now has this; task-authored geometry is still primitives).
- Add a batch/sweep mode output (CSV or similar) for larger randomized
  scenario studies.
- Consider a Cartesian-space (straight-line end-effector) waypoint mode for
  the short descend/lift segments, for cluttered scenes where even those
  need obstacle awareness.
- Analytic (closed-form) 6-DOF IK for this arm, to enumerate its actual
  finite set of solution branches directly instead of hoping bounded random
  restarts land near a collision-free one (see the transit planner's
  known limitation above).
