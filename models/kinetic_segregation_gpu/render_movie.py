#!/usr/bin/env python3
"""Render a movie from KS simulation frame dumps (2-panel: molecules + height)."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Auto-detect ffmpeg from the conda env bin directory (may not be on PATH).
if shutil.which("ffmpeg") is None:
    _env_ffmpeg = Path(sys.executable).parent / "ffmpeg"
    if _env_ffmpeg.exists():
        matplotlib.rcParams["animation.ffmpeg_path"] = str(_env_ffmpeg)


PATCH_SIZE_NM = 2000.0


def load_frame(frames_dir: Path, step: int, grid_size: int, n_tcr: int, n_cd45: int):
    """Load height field and molecule positions for a given step."""
    h = np.fromfile(frames_dir / f"h_{step:05d}.bin", dtype=np.float32)
    h = h.reshape(grid_size, grid_size)

    mol = np.fromfile(frames_dir / f"mol_{step:05d}.bin", dtype=np.float64)
    tcr_pos = mol[: n_tcr * 2].reshape(n_tcr, 2)
    cd45_pos = mol[n_tcr * 2 :].reshape(n_cd45, 2)
    return h, tcr_pos, cd45_pos


def _compute_depletion_width(tcr_pos, cd45_pos, patch_size):
    center = patch_size / 2.0
    tcr_r = np.sqrt(np.sum((tcr_pos - center) ** 2, axis=1))
    cd45_r = np.sqrt(np.sum((cd45_pos - center) ** 2, axis=1))
    return max(0.0, float(np.median(cd45_r) - np.median(tcr_r)))


def main():
    parser = argparse.ArgumentParser(description="Render KS simulation movie")
    parser.add_argument("frames_dir", type=str, help="Path to frames directory")
    parser.add_argument("-o", "--output", type=str, default="ks_movie.mp4",
                        help="Output file (default: ks_movie.mp4)")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second")
    parser.add_argument("--dpi", type=int, default=120, help="Resolution")
    parser.add_argument("--skip", type=int, default=1, help="Frame skip (use every Nth frame)")
    parser.add_argument("--rigidity", type=float, default=None,
                        help="Override rigidity label (kT/nm2), otherwise read from meta.json")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    meta = json.loads((frames_dir / "meta.json").read_text())
    grid_size = meta["grid_size"]
    n_tcr = meta["n_tcr"]
    n_cd45 = meta["n_cd45"]
    n_steps = meta["n_steps"]
    patch_nm = meta.get("patch_nm", PATCH_SIZE_NM)
    dump_interval = meta.get("dump_interval", 1)
    dt = meta.get("dt", 0.0)

    rigidity = args.rigidity if args.rigidity is not None else meta.get("rigidity_kT_nm2")
    n_pmhc = meta.get("n_pmhc", 0)
    n_frames = meta.get("n_frames", n_steps)
    all_frames = list(range(0, n_frames + 1))
    steps = all_frames[::args.skip]

    # Load static pMHC positions if present
    pmhc_pos = None
    pmhc_path = frames_dir / "pmhc.bin"
    if n_pmhc > 0 and pmhc_path.exists():
        pmhc_pos = np.fromfile(pmhc_path, dtype=np.float64).reshape(n_pmhc, 2)

    print(f"Rendering {len(steps)} frames (grid={grid_size}, n_steps={n_steps}, "
          f"dump_interval={dump_interval}, skip={args.skip}, n_pmhc={n_pmhc})")

    # Load first frame
    h0, tcr0, cd450 = load_frame(frames_dir, 0, grid_size, n_tcr, n_cd45)

    center = patch_nm / 2.0
    extent_um = [0, patch_nm / 1000, 0, patch_nm / 1000]

    fig, (ax_mol, ax_h) = plt.subplots(1, 2, figsize=(14, 6))
    fig.set_facecolor("white")

    # Left panel: molecules with depletion zone
    ax_mol.set_xlim(0, patch_nm / 1000)
    ax_mol.set_ylim(0, patch_nm / 1000)
    ax_mol.set_aspect("equal")
    ax_mol.set_xlabel("x (um)")
    ax_mol.set_ylabel("y (um)")

    tcr_scat = ax_mol.scatter([], [], c="red", s=20, alpha=0.7, label="TCR", zorder=3)
    cd45_scat = ax_mol.scatter([], [], c="royalblue", s=12, alpha=0.5, label="CD45", zorder=2)
    if pmhc_pos is not None:
        pmhc_um = pmhc_pos / 1000.0
        ax_mol.scatter(pmhc_um[:, 0], pmhc_um[:, 1], c="black", marker="x",
                       s=30, alpha=0.6, label="pMHC", zorder=1, linewidths=1)
    c_um = center / 1000.0
    depl_inner = plt.Circle((c_um, c_um), 0, fill=False, color="gold", lw=2, ls="--", zorder=4)
    depl_outer = plt.Circle((c_um, c_um), 0, fill=False, color="gold", lw=2, ls="--", zorder=4)
    ax_mol.add_patch(depl_inner)
    ax_mol.add_patch(depl_outer)
    ax_mol.legend(loc="upper right", fontsize=9)
    title_text = ax_mol.set_title("t = 0.00s")

    # Right panel: membrane height
    im = ax_h.imshow(h0.T, cmap="viridis", origin="lower", extent=extent_um,
                     vmin=0, vmax=50, aspect="equal")
    ax_h.set_xlabel("x (um)")
    ax_h.set_ylabel("y (um)")
    ax_h.set_title("Membrane height (nm)")
    fig.colorbar(im, ax=ax_h, label="h (nm)", shrink=0.8)
    if rigidity is not None:
        fig.suptitle(f"Membrane rigidity: {rigidity:g} kT/nm\u00b2", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95] if rigidity is not None else [0, 0, 1, 1])

    def update(frame_idx):
        fidx = steps[frame_idx]
        h, tcr, cd45 = load_frame(frames_dir, fidx, grid_size, n_tcr, n_cd45)

        # Convert to um for scatter
        tcr_um = tcr / 1000.0
        cd45_um = cd45 / 1000.0
        tcr_scat.set_offsets(tcr_um)
        cd45_scat.set_offsets(cd45_um)

        # Depletion circles
        tcr_r = np.sqrt(np.sum((tcr_um - c_um) ** 2, axis=1))
        cd45_r = np.sqrt(np.sum((cd45_um - c_um) ** 2, axis=1))
        depl_inner.set_radius(np.percentile(tcr_r, 75))
        depl_outer.set_radius(np.percentile(cd45_r, 25))

        # Height
        im.set_data(h.T)

        # Title with physical time and depletion
        sim_step = fidx * dump_interval
        t_phys = sim_step * dt if dt > 0 else 0
        depl_w = _compute_depletion_width(tcr, cd45, patch_nm)
        title_text.set_text(f"t = {t_phys:.3f}s  |  depletion = {depl_w:.0f} nm")

        return tcr_scat, cd45_scat, im, title_text

    ani = animation.FuncAnimation(fig, update, frames=len(steps),
                                  interval=1000 // args.fps, blit=False)

    output = args.output
    print(f"Saving to {output} ...")
    if output.endswith(".gif"):
        ani.save(output, writer="pillow", fps=args.fps, dpi=args.dpi)
    else:
        writer = animation.FFMpegWriter(fps=args.fps, extra_args=["-pix_fmt", "yuv420p"])
        ani.save(output, writer=writer, dpi=args.dpi)
    print(f"Done: {output}")
    plt.close()


if __name__ == "__main__":
    main()
