# -*- coding: utf-8 -*-
"""
Batch crop fixed ROI from images, resize to target size, save under <input_dir>/<out_subdir>/.

Default ROI: x=900, y=700, w=768, h=768 -> resize to 512x512.

Interpolation (auto by default):
  - 原图：LANCZOS4（BGR）
  - 标注：INTER_NEAREST（文件名含 mask/label/ann/seg/gt 等关键词，或整目录 --mode label）
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

# 文件名（不含路径、小写）命中这些子串则视为标注图，强制 NEAREST + 按单通道处理
_LABEL_STEM_KEYWORDS = (
    'mask', 'label', 'labels', 'ann', 'anno', 'seg', 'gt', 'semantic',
)
#  file 顶部常量，改这里即可
DEFAULT_INPUT_DIR = Path(r"C:\Users\ASUS\Desktop\DL12-NG\NG")


def looks_like_label_file(filename: str) -> bool:
    stem = Path(filename).stem.lower()
    return any(k in stem for k in _LABEL_STEM_KEYWORDS)


def crop_resize(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    out_w: int,
    out_h: int,
    interpolation: int,
) -> tuple[np.ndarray, bool]:
    """
    Crop [y:y+h, x:x+w] with bounds clamping, then resize to (out_w, out_h).
    ``img`` may be HxW (uint8) or HxWxC.
    """
    H, W = img.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    clipped = (x1 != x) or (y1 != y) or (x2 != x + w) or (y2 != y + h)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f'Crop region empty or fully outside image; image size=({W},{H}), '
            f'requested roi=({x},{y},{w},{h})')

    patch = img[y1:y2, x1:x2].copy()
    channels = 1 if patch.ndim == 2 else patch.shape[2]

    if patch.shape[0] != h or patch.shape[1] != w:
        if channels == 1:
            canvas = np.zeros((h, w), dtype=patch.dtype)
        else:
            canvas = np.zeros((h, w, channels), dtype=patch.dtype)
        dy = y1 - y
        dx = x1 - x
        ph, pw = patch.shape[:2]
        canvas[dy : dy + ph, dx : dx + pw] = patch
        patch = canvas

    resized = cv2.resize(patch, (out_w, out_h), interpolation=interpolation)
    return resized, clipped


def load_as_bgr(path: Path) -> np.ndarray | None:
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def load_as_label(path: Path) -> np.ndarray | None:
    """Keep single channel if possible; otherwise UNCHANGED (e.g. palette PNG)."""
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 3:
        # 少见：假彩色标注，仍走最近邻
        return img
    if img.ndim == 3 and img.shape[2] == 4:
        return img[..., 0]
    return img


def main() -> None:
    p = argparse.ArgumentParser(
        description='Crop ROI from all images in folder, resize, save to <folder>/<out_subdir>/.')
    p.add_argument(
        '--input-dir', '-i',
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help='...')
    p.add_argument('--x', type=int, default=0, help='Crop left')
    p.add_argument('--y', type=int, default=0, help='Crop top')
    p.add_argument('--w', type=int, default=2048, help='Crop width')
    p.add_argument('--h', type=int, default=2048, help='Crop height')
    p.add_argument('--out-w', type=int, default=1024, help='Output width')
    p.add_argument('--out-h', type=int, default=1024, help='Output height')
    p.add_argument(
        '--out-subdir',
        type=str,
        default='crop',
        help='Subdirectory under input_dir to write results')
    p.add_argument(
        '--mode',
        choices=('auto', 'rgb', 'label'),
        default='rgb',
        help='auto: NEAREST if filename looks like label; else LANCZOS4. '
             'rgb/label: force all files.',
    )
    args = p.parse_args()

    indir = args.input_dir.resolve()
    if not indir.is_dir():
        raise SystemExit(f'Not a directory: {indir}')

    outdir = indir / args.out_subdir
    outdir.mkdir(parents=True, exist_ok=True)

    names = sorted(
        f for f in os.listdir(indir)
        if Path(f).suffix.lower() in IMG_EXTS
        and (indir / f).is_file()
    )
    if not names:
        print(f'No images found in {indir}')
        return

    ok, clipped_n = 0, 0
    for name in names:
        path = indir / name
        if args.mode == 'label':
            is_label = True
        elif args.mode == 'rgb':
            is_label = False
        else:
            is_label = looks_like_label_file(name)

        if is_label:
            img = load_as_label(path)
            interp = cv2.INTER_NEAREST
        else:
            img = load_as_bgr(path)
            interp = cv2.INTER_LANCZOS4

        if img is None:
            print(f'Skip (read fail): {name}')
            continue

        try:
            out, clipped = crop_resize(
                img,
                args.x,
                args.y,
                args.w,
                args.h,
                args.out_w,
                args.out_h,
                interpolation=interp,
            )
        except ValueError as e:
            print(f'Skip {name}: {e}')
            continue
        if clipped:
            clipped_n += 1
            print(f'Clamped ROI for {name} (image smaller than ROI or negative origin)')

        dst = outdir / name
        ext = dst.suffix.lower()
        if ext in {'.jpg', '.jpeg'}:
            ret, buf = cv2.imencode(ext, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            ret, buf = cv2.imencode(ext if ext else '.png', out)
        if not ret:
            print(f'Encode fail: {name}')
            continue
        buf.tofile(str(dst))
        ok += 1

    print(f'Done. Wrote {ok} files to {outdir} ({clipped_n} with boundary clamp/pad).')


if __name__ == '__main__':
    main()
