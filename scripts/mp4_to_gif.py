#!/usr/bin/env python3
"""
Convert one or more of runner.py's --video MP4 outputs (e.g. the files in
renders/) into GIFs -- mainly so a demo can be embedded in README.md (or
anywhere else that doesn't render inline <video>) without needing a click
to play.

Shells out to ffmpeg rather than pulling in a Python video-decoding
dependency, using the standard two-pass palette approach
(palettegen -> paletteuse) instead of a naive single-pass conversion:
ffmpeg's default GIF encoder only gets a fixed 256-color web-safe-ish
palette, which bands and dithers badly on the kind of soft gradient
lighting MuJoCo's offscreen renderer produces. Generating a palette from
the actual clip first keeps the color banding down substantially, at the
cost of decoding the input twice.

Run:
    python3 scripts/mp4_to_gif.py ../renders/kr35_obstacle_demo.mp4
    python3 scripts/mp4_to_gif.py ../renders/kr35_obstacle_demo.mp4 -o out.gif --fps 15 --width 640
    python3 scripts/mp4_to_gif.py ../renders/                      # convert every *.mp4 in a directory
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def convert_one(src: Path, dst: Path, fps: int, width: int, start: float | None, duration: float | None) -> None:
    """Convert a single MP4 to a GIF via ffmpeg's palettegen/paletteuse pair.

    palettegen and paletteuse run as two filter stages in one ffmpeg
    invocation, joined with ffmpeg's filter_complex graph syntax (split the
    scaled stream into two copies -- one builds the palette, the other gets
    quantized against it) rather than writing a palette PNG to a temp file
    and shelling out twice.
    """
    trim_args = []
    if start is not None:
        trim_args += ["-ss", str(start)]
    if duration is not None:
        trim_args += ["-t", str(duration)]

    # scale first (single filter shared by both branches via split), then
    # one branch builds a palette from the actual frames, the other gets
    # quantized against that palette -- this is the standard ffmpeg
    # high-quality-GIF recipe, not something bespoke to this project.
    filter_complex = (
        f"[0:v] fps={fps},scale={width}:-1:flags=lanczos,split [a][b];"
        f"[a] palettegen [p];"
        f"[b][p] paletteuse"
    )

    cmd = [
        "ffmpeg", "-y",
        *trim_args,
        "-i", str(src),
        "-filter_complex", filter_complex,
        "-loop", "0",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed converting {src}:\n{result.stderr}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="an MP4 file, or a directory to convert every *.mp4 in")
    parser.add_argument("-o", "--output", type=Path, default=None,
                         help="output GIF path (single-file mode only; default: same name, .gif extension)")
    parser.add_argument("--fps", type=int, default=12, help="output frame rate (default: 12)")
    parser.add_argument("--width", type=int, default=480,
                         help="output width in pixels, height scales to preserve aspect ratio (default: 480)")
    parser.add_argument("--start", type=float, default=None, help="trim: start offset in seconds")
    parser.add_argument("--duration", type=float, default=None, help="trim: duration in seconds")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        print("error: ffmpeg not found on PATH -- install it (e.g. apt install ffmpeg) and retry", file=sys.stderr)
        return 1

    if args.input.is_dir():
        if args.output is not None:
            print("error: -o/--output isn't supported when converting a whole directory", file=sys.stderr)
            return 1
        sources = sorted(args.input.glob("*.mp4"))
        if not sources:
            print(f"no .mp4 files found in {args.input}", file=sys.stderr)
            return 1
    else:
        sources = [args.input]

    for src in sources:
        dst = args.output if args.output is not None else src.with_suffix(".gif")
        print(f"{src} -> {dst}")
        convert_one(src, dst, args.fps, args.width, args.start, args.duration)

    return 0


if __name__ == "__main__":
    sys.exit(main())
