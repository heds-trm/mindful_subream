#!/usr/bin/env python3
"""
Parse a TIC CSV and export JPG(s) per lesion:
  <LesionID>_tic.jpg

Each JPG contains:
- TIC curve for phases 0..12 (line + square markers)
- A red line corresponding to the maximum-slope (MS) segment between two
  consecutive phases where the delta is maximal.

Features:
- LesionID header is assumed.
- --nrows allows processing only the first N rows.
- --timestep defines time (in seconds) between two consecutive phases (default=1.0).
- Matplotlib Agg backend is mandatory (offline/headless rendering).
- tqdm progress bar.
"""
import argparse
import os
import re
from typing import List, Tuple

# ---- Mandatory OFFLINE backend ----
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

PHASE_COL_RE = re.compile(r"^phase[_\s]?(\d+)$", re.IGNORECASE)


def find_phase_columns(columns: List[str]) -> List[str]:
    """Return phase columns sorted by phase index (e.g., phase_0 .. phase_12)."""
    found: List[Tuple[int, str]] = []
    for c in columns:
        m = PHASE_COL_RE.match(str(c).strip())
        if m:
            found.append((int(m.group(1)), c))
    found.sort(key=lambda t: t[0])
    return [c for _, c in found]


def compute_max_slope_pair(y: np.ndarray, x: np.ndarray) -> Tuple[int, int, float]:
    """
    For consecutive points, compute deltas and return:
      (i, i+1, slope) for the pair giving the maximum slope.
    """
    if len(y) < 2:
        return 0, 0, 0.0

    slopes = np.diff(y) / np.diff(x)
    idx = int(np.argmax(slopes))

    x1, x2 = float(x[idx]), float(x[idx + 1])
    y1, y2 = float(y[idx]), float(y[idx + 1])

    slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0.0
    return idx, idx + 1, slope


def plot_one_lesion(
    lesion_id: str,
    phases_x: np.ndarray,
    phases_y: np.ndarray,
    out_path: str,
) -> None:
    i1, i2, m = compute_max_slope_pair(phases_y, phases_x)
    x1, x2 = float(phases_x[i1]), float(phases_x[i2])
    y1, y2 = float(phases_y[i1]), float(phases_y[i2])

    # Line through the two MS points: y = m*x + b
    b = y1 - m * x1
    x_line = np.array([phases_x.min(), phases_x.max()], dtype=float)
    y_line = m * x_line + b

    plt.figure(figsize=(8, 5), dpi=150)

    # TIC curve
    plt.plot(phases_x, phases_y, marker="s", linewidth=2)

    # Highlight MS points
    plt.scatter([x1, x2], [y1, y2], marker="s", s=60)

    # MS line (red)
    plt.plot(x_line, y_line, color="red", linewidth=2)

    plt.title(f"TIC - Lesion {lesion_id}")
    plt.xlabel("Time (s)")
    plt.ylabel("Signal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_path, format="jpg")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/mnt/data/tic_data.csv", help="Input CSV path")
    ap.add_argument("--outdir", default=".", help="Output directory for JPGs")
    ap.add_argument("--nrows", type=int, default=None, help="Process only the first N rows")
    ap.add_argument(
        "--timestep",
        type=float,
        default=1.0,
        help="Time (in seconds) between two consecutive phases (default=1.0)",
    )
    args = ap.parse_args()

    if args.timestep <= 0:
        raise SystemExit("--timestep must be > 0")

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "LesionID" not in df.columns:
        raise SystemExit('Expected a "LesionID" column in the CSV header.')

    if args.nrows is not None:
        if args.nrows < 0:
            raise SystemExit("--nrows must be >= 0")
        df = df.head(args.nrows)

    phase_cols_all = find_phase_columns(list(df.columns))
    if not phase_cols_all:
        raise SystemExit("No phase columns found (expected columns like phase_0 ... phase_12).")

    # Prefer phases 0..12 if available
    wanted = set(range(13))
    phase_cols_0_12 = []
    for c in phase_cols_all:
        m = PHASE_COL_RE.match(c)
        if m and int(m.group(1)) in wanted:
            phase_cols_0_12.append(c)

    phase_cols = phase_cols_0_12 if len(phase_cols_0_12) >= 2 else phase_cols_all

    phase_indices = [int(PHASE_COL_RE.match(c).group(1)) for c in phase_cols]  # type: ignore

    # Scale X axis using timestep
    phases_x = np.array(phase_indices, dtype=float) * args.timestep

    written = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing lesions"):
        lesion_id = str(row["LesionID"])
        phases_y = row[phase_cols].to_numpy(dtype=float)

        out_path = os.path.join(args.outdir, f"{lesion_id}_tic.jpg")
        plot_one_lesion(lesion_id, phases_x, phases_y, out_path)
        written += 1


import argparse
import os
import re
from typing import List, Tuple

# ---- Mandatory OFFLINE backend ----
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

PHASE_COL_RE = re.compile(r"^phase[_\s]?(\d+)$", re.IGNORECASE)


def find_phase_columns(columns: List[str]) -> List[str]:
    """Return phase columns sorted by phase index (e.g., phase_0 .. phase_12)."""
    found: List[Tuple[int, str]] = []
    for c in columns:
        m = PHASE_COL_RE.match(str(c).strip())
        if m:
            found.append((int(m.group(1)), c))
    found.sort(key=lambda t: t[0])
    return [c for _, c in found]


def compute_max_slope_pair(y: np.ndarray, x: np.ndarray) -> Tuple[int, int, float]:
    """
    For consecutive points, compute deltas and return:
      (i, i+1, slope) for the pair giving the maximum delta.
    """
    if len(y) < 2:
        return 0, 0, 0.0

    deltas = np.diff(y)
    idx = int(np.argmax(deltas))

    x1, x2 = float(x[idx]), float(x[idx + 1])
    y1, y2 = float(y[idx]), float(y[idx + 1])

    slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0.0
    return idx, idx + 1, slope


def plot_one_lesion(lesion_id: str, phases_x: np.ndarray, phases_y: np.ndarray, out_path: str) -> None:
    i1, i2, m = compute_max_slope_pair(phases_y, phases_x)
    x1, x2 = float(phases_x[i1]), float(phases_x[i2])
    y1, y2 = float(phases_y[i1]), float(phases_y[i2])

    # Line through the two MS points: y = m*x + b
    b = y1 - m * x1
    x_line = np.array([phases_x.min(), phases_x.max()], dtype=float)
    y_line = m * x_line + b

    plt.figure(figsize=(8, 5), dpi=150)

    # TIC curve: line + square markers
    plt.plot(phases_x, phases_y, marker="s", linewidth=2)

    # Highlight MS points
    plt.scatter([x1, x2], [y1, y2], marker="s", s=60)

    # MS line (red)
    plt.plot(x_line, y_line, color="red", linewidth=2)

    plt.title(f"TIC - Lesion {lesion_id}")
    plt.xlabel("Phase")
    plt.ylabel("Signal")
    plt.xticks(phases_x.astype(int))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_path, format="jpg")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/mnt/data/tic_data.csv", help="Input CSV path")
    ap.add_argument("--outdir", default=".", help="Output directory for JPGs")
    ap.add_argument("--nrows", type=int, default=None, help="Process only the first N rows")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "LesionID" not in df.columns:
        raise SystemExit('Expected a "LesionID" column in the CSV header.')

    if args.nrows is not None:
        if args.nrows < 0:
            raise SystemExit("--nrows must be >= 0")
        df = df.head(args.nrows)

    phase_cols_all = find_phase_columns(list(df.columns))
    if not phase_cols_all:
        raise SystemExit("No phase columns found (expected columns like phase_0 ... phase_12).")

    # Prefer phases 0..12 if available
    wanted = set(range(13))
    phase_cols_0_12 = []
    for c in phase_cols_all:
        m = PHASE_COL_RE.match(c)
        if m and int(m.group(1)) in wanted:
            phase_cols_0_12.append(c)

    phase_cols = phase_cols_0_12 if len(phase_cols_0_12) >= 2 else phase_cols_all

    phase_indices = [int(PHASE_COL_RE.match(c).group(1)) for c in phase_cols]  # type: ignore
    phases_x = np.array(phase_indices, dtype=float)

    written = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing lesions"):
        lesion_id = str(row["LesionID"])
        phases_y = row[phase_cols].to_numpy(dtype=float)

        out_path = os.path.join(args.outdir, f"{lesion_id}_tic.jpg")
        plot_one_lesion(lesion_id, phases_x, phases_y, out_path)
        written += 1

    print(f"Done. Wrote {written} JPG(s) to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
