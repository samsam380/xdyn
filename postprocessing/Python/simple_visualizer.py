#!/usr/bin/env python3
"""Quick 2D visualizer for xdyn CSV outputs.

This script turns xdyn result CSV files into an animated GIF or MP4 showing:
- the trajectory (x,y)
- a moving ship marker
- optional heading arrow when a heading column is found
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd


COLUMN_CANDIDATES = {
    "x": ["x", "X", "x(m)"],
    "y": ["y", "Y", "y(m)"],
    "t": ["t", "time", "Time"],
    "psi": ["psi", "heading", "yaw"],
}


def _match_prefixed(columns, prefix):
    for name in columns:
        normalized = name.strip().lower()
        if normalized == prefix.lower() or normalized.startswith(prefix.lower() + "("):
            return name
    return None


def guess_column(columns, kind):
    if kind in {"x", "y", "t", "psi"}:
        matched = _match_prefixed(columns, kind)
        if matched is not None:
            return matched

    for candidate in COLUMN_CANDIDATES.get(kind, []):
        for name in columns:
            if name.strip().lower() == candidate.lower():
                return name
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Animate xdyn CSV trajectory in 2D")
    parser.add_argument("csv", type=Path, help="Input CSV file produced by xdyn")
    parser.add_argument("-o", "--output", default="xdyn_preview.gif", help="Output animation file")
    parser.add_argument("--x-col", help="Override X position column")
    parser.add_argument("--y-col", help="Override Y position column")
    parser.add_argument("--t-col", help="Override time column")
    parser.add_argument("--psi-col", help="Override heading/yaw column (degrees or radians)")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second")
    parser.add_argument("--step", type=int, default=1, help="Use one frame every N samples")
    parser.add_argument("--title", default="xdyn trajectory preview", help="Figure title")
    parser.add_argument(
        "--heading-in-deg",
        action="store_true",
        help="Interpret heading column as degrees (default: auto-detect)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data = pd.read_csv(args.csv)
    columns = list(data.columns)

    x_col = args.x_col or guess_column(columns, "x")
    y_col = args.y_col or guess_column(columns, "y")
    t_col = args.t_col or guess_column(columns, "t")
    psi_col = args.psi_col or guess_column(columns, "psi")

    if x_col is None or y_col is None:
        raise SystemExit(
            "Could not auto-detect x/y columns. Provide --x-col and --y-col. "
            f"Available columns: {columns}"
        )

    if args.step < 1:
        raise SystemExit("--step must be >= 1")

    sampled = data.iloc[:: args.step].reset_index(drop=True)
    x = sampled[x_col].to_numpy(dtype=float)
    y = sampled[y_col].to_numpy(dtype=float)
    t = sampled[t_col].to_numpy(dtype=float) if t_col is not None else np.arange(len(sampled))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(args.title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    xpad = max(1e-6, 0.08 * (x.max() - x.min() if len(x) > 1 else 1.0))
    ypad = max(1e-6, 0.08 * (y.max() - y.min() if len(y) > 1 else 1.0))
    ax.set_xlim(x.min() - xpad, x.max() + xpad)
    ax.set_ylim(y.min() - ypad, y.max() + ypad)

    (trail,) = ax.plot([], [], "-", lw=2, label="trajectory")
    (ship,) = ax.plot([], [], "o", ms=8, label="ship")
    heading_line = ax.plot([], [], lw=2, label="heading")[0] if psi_col is not None else None
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    arrow_length = max(1e-6, 0.06 * max(x.max() - x.min(), y.max() - y.min(), 1.0))

    psi = None
    if psi_col is not None:
        psi = sampled[psi_col].to_numpy(dtype=float)
        if not args.heading_in_deg:
            if np.nanmax(np.abs(psi)) <= 2 * math.pi + 0.1:
                psi = np.degrees(psi)
        else:
            psi = np.asarray(psi)

    def init():
        trail.set_data([], [])
        ship.set_data([], [])
        if heading_line is not None:
            heading_line.set_data([], [])
        time_text.set_text("")
        return [trail, ship, heading_line, time_text] if heading_line is not None else [trail, ship, time_text]

    def update(i):
        trail.set_data(x[: i + 1], y[: i + 1])
        ship.set_data([x[i]], [y[i]])
        if heading_line is not None and psi is not None:
            rad = math.radians(psi[i])
            hx = x[i] + arrow_length * math.cos(rad)
            hy = y[i] + arrow_length * math.sin(rad)
            heading_line.set_data([x[i], hx], [y[i], hy])
        time_text.set_text(f"t = {t[i]:.3f}")
        return [trail, ship, heading_line, time_text] if heading_line is not None else [trail, ship, time_text]

    anim = FuncAnimation(fig, update, init_func=init, frames=len(sampled), interval=1000 / max(args.fps, 1), blit=True)

    output_path = Path(args.output)
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        anim.save(output_path, writer=PillowWriter(fps=args.fps))
    else:
        anim.save(output_path, fps=args.fps)

    print(f"Saved animation to: {output_path}")


if __name__ == "__main__":
    main()
