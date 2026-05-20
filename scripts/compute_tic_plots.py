"""
Parse a TIC CSV and export JPG(s) per lesion:
  <LesionID>_tic.jpg

Max-slope (MS) rule:
- If CSV has BOTH headers "MS1phase" and "MS2phase" and the row values are usable,
  use those 2 phase indices (MS2phase > MS1phase) to compute the slope and draw the MS line.
- Otherwise compute MS automatically as the maximum slope between consecutive points.

Always write an augmented CSV in the output directory:
  <input_name>2.csv   (same name + "2" before extension)
This output CSV will contain MS1phase and MS2phase columns filled with:
- existing values if present/usable, OR
- computed values if absent/unusable.

Other features:
- LesionID header is assumed.
- --nrows allows processing only the first N rows.
- --timestep defines time (in seconds) between two consecutive phases (default=1.0).
- Matplotlib Agg backend is mandatory (offline/headless rendering).
- tqdm progress bar.
- Y-axis padding: ±5% of (y_max - y_min).
"""

import argparse
import os
import re
from typing import List, Tuple, Optional

# ---- Mandatory OFFLINE backend ----
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

PHASE_COL_RE = re.compile(r"^phase[_\s]?(\d+)$", re.IGNORECASE)


def find_phase_columns(columns: List[str]) -> List[str]:
    found: List[Tuple[int, str]] = []
    for c in columns:
        m = PHASE_COL_RE.match(str(c).strip())
        if m:
            found.append((int(m.group(1)), c))
    found.sort(key=lambda t: t[0])
    return [c for _, c in found]


def compute_ms_auto(y: np.ndarray, x: np.ndarray) -> Tuple[int, int, float]:
    if len(y) < 2:
        return 0, 0, 0.0
    slopes = np.diff(y) / np.diff(x)
    i1 = int(np.argmax(slopes))
    i2 = i1 + 1
    slope = float(slopes[i1])
    return i1, i2, slope


def compute_ms_from_phases(
    y: np.ndarray,
    x: np.ndarray,
    phase_indices: List[int],
    ms1phase: int,
    ms2phase: int,
) -> Optional[Tuple[int, int, float]]:
    if ms2phase <= ms1phase:
        return None

    if ms1phase not in phase_indices or ms2phase not in phase_indices:
        return None

    i1 = phase_indices.index(ms1phase)
    i2 = phase_indices.index(ms2phase)

    if i1 == i2:
        return None

    slope = float((y[i2] - y[i1]) / (x[i2] - x[i1]))
    if not np.isfinite(slope):
        return None

    return i1, i2, slope


def plot_one_lesion(
    lesion_id: str,
    x: np.ndarray,
    y: np.ndarray,
    i1: int,
    i2: int,
    slope: float,
    out_path: str,
) -> None:

    x1, x2 = float(x[i1]), float(x[i2])
    y1, y2 = float(y[i1]), float(y[i2])

    b = y1 - slope * x1
    x_line = np.array([float(x.min()), float(x.max())])
    y_line = slope * x_line + b

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_range = y_max - y_min

    if y_range > 0:
        pad = 0.05 * y_range
        y_lower = y_min - pad
        y_upper = y_max + pad
    else:
        y_lower = y_min - 1.0
        y_upper = y_max + 1.0

    plt.figure(figsize=(8, 5), dpi=150)

    plt.plot(x, y, marker="s", linewidth=2)
    plt.scatter([x1, x2], [y1, y2], marker="s", s=60)
    plt.plot(x_line, y_line, color="red", linewidth=2)

    plt.title(f"TIC - Lesion {lesion_id}")
    plt.xlabel("Time (s)")
    plt.ylabel("Signal")
    plt.ylim(y_lower, y_upper)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_path, format="jpg")
    plt.close()


def augmented_csv_path(input_csv: str, outdir: str) -> str:
    base = os.path.basename(input_csv)
    root, ext = os.path.splitext(base)
    if ext.lower() != ".csv":
        ext = ".csv"
    return os.path.join(outdir, f"{root}2{ext}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/mnt/data/tic_data.csv")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--timestep", type=float, default=1.0)
    args = ap.parse_args()

    if args.timestep <= 0:
        raise SystemExit("--timestep must be > 0")

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)

    if "LesionID" not in df.columns:
        raise SystemExit('Expected a "LesionID" column.')

    if args.nrows is not None:
        df = df.head(args.nrows).copy()
    else:
        df = df.copy()

    # Ensure MS columns exist
    has_ms_cols = ("MS1phase" in df.columns) and ("MS2phase" in df.columns)
    if "MS1phase" not in df.columns:
        df["MS1phase"] = np.nan
    if "MS2phase" not in df.columns:
        df["MS2phase"] = np.nan

    phase_cols = find_phase_columns(list(df.columns))
    if not phase_cols:
        raise SystemExit("No phase columns found.")

    phase_indices = [int(PHASE_COL_RE.match(c).group(1)) for c in phase_cols]
    x = np.array(phase_indices, dtype=float) * args.timestep

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing lesions"):
        lesion_id = str(row["LesionID"])
        y = row[phase_cols].to_numpy(dtype=float)

        ms_pair = None

        if has_ms_cols:
            try:
                ms1p = int(row["MS1phase"])
                ms2p = int(row["MS2phase"])
                ms_pair = compute_ms_from_phases(
                    y, x, phase_indices, ms1p, ms2p
                )
            except Exception:
                ms_pair = None

        if ms_pair is not None:
            i1, i2, slope = ms_pair
            ms1_fill = phase_indices[i1]
            ms2_fill = phase_indices[i2]
        else:
            i1, i2, slope = compute_ms_auto(y, x)
            ms1_fill = phase_indices[i1]
            ms2_fill = phase_indices[i2]

        df.at[idx, "MS1phase"] = ms1_fill
        df.at[idx, "MS2phase"] = ms2_fill

        out_path = os.path.join(args.outdir, f"{lesion_id}_tic.jpg")
        plot_one_lesion(lesion_id, x, y, i1, i2, slope, out_path)

    out_csv = augmented_csv_path(args.csv, args.outdir)
    df.to_csv(out_csv, index=False)

    print(f"Wrote {len(df)} JPG(s)")
    print(f"Augmented CSV: {out_csv}")


if __name__ == "__main__":
    main()
