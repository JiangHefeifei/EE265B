#!/usr/bin/env python3
"""Assemble a horizontal frame strip from a rollout video for the report figures.

Usage examples:
  # uniformly sample 7 frames, label with step indices
  python make_strips.py ep01.mp4 --frames auto7 --labels steps -o fig_task.png

  # specific frame indices with per-frame caption text (use | to separate)
  python make_strips.py ep12.mp4 --frames 300,380,460,540 \
      --captions "pick1|pick1 (stale)|pick1 (stale)|pick2" -o fig_case_ep12.png

Requires: pip install opencv-python pillow
"""
import argparse

import cv2
from PIL import Image, ImageDraw, ImageFont


def load_frames(video_path, spec):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if spec.startswith("auto"):
        n = int(spec[4:] or 7)
        idxs = [int(round(i * (total - 1) / (n - 1))) for i in range(n)]
    else:
        idxs = [int(x) for x in spec.split(",")]
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"cannot read frame {i} (video has {total})")
        frames.append((i, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frames", default="auto7", help="auto<N> or comma-separated frame indices")
    ap.add_argument("--labels", choices=["steps", "none"], default="steps",
                    help="print 't = <idx>' under each frame")
    ap.add_argument("--captions", default=None,
                    help="per-frame caption text, separated by | (e.g. subgoal strings)")
    ap.add_argument("--height", type=int, default=256, help="per-frame height in px")
    ap.add_argument("--pad", type=int, default=6, help="gap between frames in px")
    ap.add_argument("-o", "--out", default="strip.png")
    args = ap.parse_args()

    frames = load_frames(args.video, args.frames)
    captions = args.captions.split("|") if args.captions else None
    if captions and len(captions) != len(frames):
        raise SystemExit(f"{len(captions)} captions for {len(frames)} frames")

    h = args.height
    scaled = [(i, im.resize((int(im.width * h / im.height), h))) for i, im in frames]
    text_h = (22 if args.labels == "steps" else 0) + (26 if captions else 0)
    W = sum(im.width for _, im in scaled) + args.pad * (len(scaled) - 1)
    H = h + text_h
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
        font_b = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = font_b = ImageFont.load_default()

    x = 0
    for k, (idx, im) in enumerate(scaled):
        canvas.paste(im, (x, 0))
        cy = h + 2
        if args.labels == "steps":
            draw.text((x + im.width // 2, cy), f"t = {idx}", fill="black", font=font, anchor="ma")
            cy += 22
        if captions:
            stale = "stale" in captions[k].lower()
            draw.text((x + im.width // 2, cy), captions[k],
                      fill="red" if stale else "black",
                      font=font_b if stale else font, anchor="ma")
        x += im.width + args.pad

    canvas.save(args.out)
    print(f"saved {args.out}  ({W}x{H}, {len(scaled)} frames)")


if __name__ == "__main__":
    main()
