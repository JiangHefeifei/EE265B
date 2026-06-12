#!/usr/bin/env python3
"""Create report-ready frame strips from a rollout video."""

from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_frames(spec: str, total_frames: int) -> list[int]:
    if spec.startswith("auto"):
        count = int(spec.removeprefix("auto") or "5")
        if count <= 1:
            return [0]
        start = max(0, int(total_frames * 0.08))
        end = max(start, int(total_frames * 0.94) - 1)
        return [int(round(x)) for x in np.linspace(start, end, count)]

    frames = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        frames.append(int(part))
    return frames


def read_frame(video_path: Path, frame_idx: int) -> Image.Image:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = min(max(frame_idx, 0), max(total - 1, 0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def crop_image(image: Image.Image, crop: str | None) -> Image.Image:
    if not crop:
        return image
    parts = [int(p.strip()) for p in crop.split(",")]
    if len(parts) != 4:
        raise ValueError("--crop must be formatted as left,top,right,bottom")
    left, top, right, bottom = parts
    if right <= 0:
        right = image.width + right
    if bottom <= 0:
        bottom = image.height + bottom
    left = min(max(left, 0), image.width - 1)
    top = min(max(top, 0), image.height - 1)
    right = min(max(right, left + 1), image.width)
    bottom = min(max(bottom, top + 1), image.height)
    return image.crop((left, top, right, bottom))


def fit_image(image: Image.Image, width: int) -> Image.Image:
    scale = width / image.width
    height = int(round(image.height * scale))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def wrap_caption(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current: list[str] = []
    probe = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(probe)
    for word in words:
        trial = " ".join([*current, word])
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def make_strip(
    video_path: Path,
    frame_indices: list[int],
    captions: list[str],
    output_path: Path,
    *,
    title: str = "",
    frame_width: int = 240,
    gap: int = 12,
    caption_height: int = 64,
    crop: str | None = None,
) -> None:
    if len(captions) != len(frame_indices):
        raise ValueError(f"Need {len(frame_indices)} captions, got {len(captions)}")

    frames = [fit_image(crop_image(read_frame(video_path, idx), crop), frame_width) for idx in frame_indices]
    frame_height = max(frame.height for frame in frames)
    title_height = 42 if title else 0
    width = len(frames) * frame_width + (len(frames) + 1) * gap
    height = title_height + gap + frame_height + caption_height + gap

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(22, bold=True)
    caption_font = load_font(16, bold=True)
    index_font = load_font(13, bold=False)

    if title:
        draw.text((gap, 10), title, fill=(28, 28, 28), font=title_font)

    y0 = title_height + gap
    for i, (frame, frame_idx, caption) in enumerate(zip(frames, frame_indices, captions)):
        x0 = gap + i * (frame_width + gap)
        y_img = y0 + (frame_height - frame.height) // 2
        canvas.paste(frame, (x0, y_img))
        draw.rectangle((x0, y_img, x0 + frame_width - 1, y_img + frame.height - 1), outline=(70, 70, 70), width=1)
        draw.text((x0 + 7, y_img + 6), f"f{frame_idx}", fill=(255, 255, 255), font=index_font, stroke_width=2, stroke_fill=(0, 0, 0))

        is_stale = "stale" in caption.lower()
        color = (190, 32, 32) if is_stale else (32, 32, 32)
        lines = wrap_caption(caption, caption_font, frame_width - 12)
        y_text = y0 + frame_height + 10
        for line in lines[:2]:
            bbox = draw.textbbox((0, 0), line, font=caption_font)
            draw.text((x0 + (frame_width - (bbox[2] - bbox[0])) // 2, y_text), line, fill=color, font=caption_font)
            y_text += 20

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--frames", required=True, help="Comma-separated frame ids or autoN, e.g. auto5")
    parser.add_argument("--captions", required=True, help="Captions separated by |")
    parser.add_argument("--title", default="")
    parser.add_argument("--width", type=int, default=240, help="Per-frame output width")
    parser.add_argument("--crop", default=None, help="Optional crop box: left,top,right,bottom; non-positive right/bottom are relative to image edge")
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frame_indices = parse_frames(args.frames, total_frames)
    captions = args.captions.split("|")
    make_strip(args.video, frame_indices, captions, args.output, title=args.title, frame_width=args.width, crop=args.crop)


if __name__ == "__main__":
    main()
