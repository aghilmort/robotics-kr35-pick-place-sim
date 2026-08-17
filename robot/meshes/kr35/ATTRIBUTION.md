# Mesh attribution

The STL files in `visual/` and `collision/` are derived from KUKA's real
CAD-exported meshes for the KR35 R1840-3 (CyberTech family), published by
KUKA/kroshu here:

  https://github.com/kroshu/kuka_robot_descriptions
  kuka_cybertech_support/meshes/kr35_r1840_3_hw/

Original license: Apache-2.0 (see the source repo's `LICENSE` /
`package.xml`). This project redistributes them under the same terms.

## What changed from the source

- `collision/*.stl` -- copied byte-for-byte, unmodified.
- `visual/*.dae` (Collada) -- converted to `.stl` with `trimesh` (MuJoCo's
  native mesh loader accepts STL/OBJ/MSH, not Collada). Geometry is
  unchanged; per-submesh materials/textures embedded in the .dae were not
  preserved through the conversion -- this project paints all link meshes a
  uniform KUKA-orange in the MJCF instead (see `KUKA_ORANGE` in
  `build_kr35_mjcf.py`), consistent with how the project already handled
  color before real meshes were available.

## Placement

Each mesh's pose within its own link body (`pos`/`quat` on the MuJoCo
`<geom>`) is copied verbatim from the source xacro's
`<visual>`/`<collision><origin rpy=".." xyz=".."/>` -- see `MESH_ORIGINS` in
`build_kr35_mjcf.py`. These are not zero (unlike the KR16 mesh set this
project used earlier): the source meshes are authored in a shared reference
frame from the CAD assembly, and the per-link origin is what re-composes
them correctly under forward kinematics.
