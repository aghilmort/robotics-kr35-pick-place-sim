#!/usr/bin/env python3
"""CLI entry point: validate a scenario YAML against the schema, build the
MuJoCo scene, run the scripted pick-and-place episode, and print a JSON
report.

Usage:
  python3 runner.py ../schema/examples/basic_pick_place.yaml
  python3 runner.py ../schema/examples/basic_pick_place.yaml --trials 20 --seed-from 0
  python3 runner.py ../schema/examples/basic_pick_place.yaml --video out.mp4
"""
import argparse
import copy
import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "glfw")  # headless offscreen rendering for --video (egl on Linux)

import jsonschema
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import scene_builder
import trajectory

SCHEMA_PATH = pathlib.Path(__file__).parent.parent / "schema" / "scenario.schema.json"


def load_scenario(path):
    doc = yaml.safe_load(open(path))
    schema = json.load(open(SCHEMA_PATH))
    jsonschema.validate(doc, schema)
    return doc


def run_once(doc, video_path=None):
    model, data, ctx = scene_builder.build_model(doc)
    frames = [] if video_path else None
    renderer = None
    on_frame = None

    if video_path:
        import mujoco
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.lookat = [1.0, 0.1, 0.9]
        cam.distance = 4.2
        cam.azimuth = 280
        cam.elevation = -25
        frame_every = max(1, int(round(1.0 / 30.0 / model.opt.timestep)))
        counter = {"n": 0}

        def on_frame(d):
            counter["n"] += 1
            if counter["n"] % frame_every == 0:
                renderer.update_scene(d, camera=cam)
                frames.append(renderer.render().copy())

    report = trajectory.run_episode(model, data, ctx, doc, on_frame=on_frame)

    if video_path and frames:
        import imageio
        imageio.mimsave(video_path, frames, fps=30)
        report["video"] = str(video_path)

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", help="Path to a scenario YAML file")
    ap.add_argument("--trials", type=int, default=1, help="Number of randomized trials to run")
    ap.add_argument("--seed-from", type=int, default=0, help="First seed used when --trials > 1")
    ap.add_argument("--video", type=str, default=None, help="Write an MP4 of the (single) run")
    args = ap.parse_args()

    doc = load_scenario(args.scenario)

    if args.trials == 1:
        report = run_once(doc, video_path=args.video)
        print(json.dumps(report, indent=2))
        sys.exit(0 if report["success"] else 1)

    results = []
    for i in range(args.trials):
        trial_doc = copy.deepcopy(doc)
        trial_doc["seed"] = args.seed_from + i
        results.append(run_once(trial_doc))

    n_success = sum(1 for r in results if r["success"])
    summary = {
        "scenario": doc["name"],
        "trials": args.trials,
        "successes": n_success,
        "success_rate": n_success / args.trials,
        "failure_reasons": {},
        "results": results,
    }
    for r in results:
        if not r["success"]:
            summary["failure_reasons"][r["failure_reason"]] = summary["failure_reasons"].get(r["failure_reason"], 0) + 1
    print(json.dumps(summary, indent=2))
    sys.exit(0 if n_success == args.trials else 1)


if __name__ == "__main__":
    main()
