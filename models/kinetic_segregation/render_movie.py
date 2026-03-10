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
from matplotlib.patches import Annulus
import numpy as np

# Auto-detect ffmpeg from the conda env bin directory (may not be on PATH).
if shutil.which("ffmpeg") is None:
    _env_ffmpeg = Path(sys.executable).parent / "ffmpeg"
    if _env_ffmpeg.exists():
        matplotlib.rcParams["animation.ffmpeg_path"] = str(_env_ffmpeg)

# ── Physical constants (fallbacks when not in meta.json) ─────────────
PATCH_SIZE_NM = 2000.0
INIT_HEIGHT_NM = 70.0

# ── Paul Tol bright palette (colorblind-safe) ────────────────────────
COLOR_TCR = "#EE6677"       # rose — warm, prominent
COLOR_CD45 = "#4477AA"      # blue — cool, easily separated from rose
COLOR_PMHC = "#228833"      # green — distinct from rose TCR and blue CD45
COLOR_DEPLETION = "#CCBB44"  # muted gold — annotation


def _even_figsize(w_in, h_in, dpi):
    """Round figure size so pixel dimensions are even (h264 requirement)."""
    w_px = round(w_in * dpi)
    h_px = round(h_in * dpi)
    if w_px % 2:
        w_px += 1
    if h_px % 2:
        h_px += 1
    return w_px / dpi, h_px / dpi


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
    parser.add_argument("--dpi", type=int, default=150, help="Resolution")
    parser.add_argument("--skip", type=int, default=1, help="Frame skip (use every Nth frame)")
    parser.add_argument("--rigidity", type=float, default=None,
                        help="Override rigidity label (kT/nm2), otherwise read from meta.json")
    parser.add_argument("--show-pmhc", action=argparse.BooleanOptionalAction,
                        default=True, help="Show/hide static pMHC markers (default: show)")
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
    h_vmax = meta.get("init_height", INIT_HEIGHT_NM)

    rigidity = args.rigidity if args.rigidity is not None else meta.get("rigidity_kT_nm2")
    n_pmhc = meta.get("n_pmhc", 0)
    n_frames = meta.get("n_frames", n_steps)
    all_frames = list(range(0, n_frames + 1))
    steps = all_frames[::args.skip]

    # Load static pMHC positions if present and requested
    pmhc_pos = None
    pmhc_path = frames_dir / "pmhc.bin"
    if args.show_pmhc and n_pmhc > 0 and pmhc_path.exists():
        pmhc_pos = np.fromfile(pmhc_path, dtype=np.float64).reshape(n_pmhc, 2)

    print(f"Rendering {len(steps)} frames (grid={grid_size}, n_steps={n_steps}, "
          f"dump_interval={dump_interval}, skip={args.skip}, n_pmhc={n_pmhc})")

    # Load first frame
    h0, tcr0, cd450 = load_frame(frames_dir, 0, grid_size, n_tcr, n_cd45)

    # ── Matplotlib style ─────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
    })

    center = patch_nm / 2.0
    extent_um = [0, patch_nm / 1000, 0, patch_nm / 1000]

    # Ensure even pixel dimensions for h264 codec.
    # Use figsize that produces even pixel counts at the target DPI.
    fig_w, fig_h = _even_figsize(13, 6, args.dpi)
    fig, (ax_mol, ax_h) = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    fig.set_layout_engine('none')

    # Fixed layout — never use tight_layout (causes per-frame jitter in animations)
    if rigidity is not None:
        fig.subplots_adjust(left=0.06, right=0.95, bottom=0.10, top=0.88,
                            wspace=0.28)
    else:
        fig.subplots_adjust(left=0.06, right=0.95, bottom=0.10, top=0.93,
                            wspace=0.28)

    # ── Left panel: molecules with depletion zone ────────────────────
    ax_mol.set_xlim(0, patch_nm / 1000)
    ax_mol.set_ylim(0, patch_nm / 1000)
    ax_mol.set_aspect("equal")
    ax_mol.set_xlabel("x (\u00b5m)")
    ax_mol.set_ylabel("y (\u00b5m)")

    # Draw pMHC first (background), then CD45, then TCR on top
    if pmhc_pos is not None:
        pmhc_um = pmhc_pos / 1000.0
        ax_mol.scatter(pmhc_um[:, 0], pmhc_um[:, 1], c=COLOR_PMHC, marker="x",
                       s=30, alpha=0.7, label="pMHC", zorder=1, linewidths=1.0)

    cd45_scat = ax_mol.scatter([], [], c=COLOR_CD45, s=12, alpha=0.5,
                               label="CD45", zorder=2)
    tcr_scat = ax_mol.scatter([], [], c=COLOR_TCR, s=20, alpha=0.7,
                              label="TCR", zorder=3)

    c_um = center / 1000.0
    depl_annulus = Annulus(
        (c_um, c_um), r=0.001, width=0.001,
        facecolor=COLOR_DEPLETION, alpha=0.12,
        edgecolor=COLOR_DEPLETION, linewidth=1.0,
        zorder=4,
    )
    ax_mol.add_patch(depl_annulus)
    ax_mol.legend(loc="upper right", fontsize=9)
    title_text = ax_mol.set_title("", fontsize=11, pad=8)
    ax_mol.text(0.02, 0.97, "(a)", transform=ax_mol.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="left")

    # ── Right panel: membrane height ─────────────────────────────────
    im = ax_h.imshow(h0.T, cmap="viridis", origin="lower", extent=extent_um,
                     vmin=0, vmax=h_vmax, aspect="equal")
    ax_h.set_xlabel("x (\u00b5m)")
    ax_h.set_ylabel("y (\u00b5m)")
    ax_h.set_title("Membrane height (nm)", fontsize=11, pad=8)
    cbar = fig.colorbar(im, ax=ax_h, label="h (nm)", shrink=0.82, pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(0.5)
    ax_h.text(0.02, 0.97, "(b)", transform=ax_h.transAxes,
              fontsize=13, fontweight="bold", va="top", ha="left")

    # ── Suptitle and progress bar ────────────────────────────────────
    if rigidity is not None:
        fig.suptitle(f"Membrane rigidity: {rigidity:g} kT/nm\u00b2",
                     fontsize=13, fontweight="bold")

    # Thin progress track at the bottom of the figure
    ax_prog = fig.add_axes([0.05, 0.012, 0.9, 0.012])
    ax_prog.set_xlim(0, 1)
    ax_prog.set_ylim(0, 1)
    ax_prog.axis("off")
    ax_prog.axhline(0.5, color="0.85", linewidth=3, solid_capstyle="round")
    progress_line, = ax_prog.plot([0], [0.5], color=COLOR_TCR, linewidth=3,
                                  solid_capstyle="round")

    def update(frame_idx):
        fidx = steps[frame_idx]
        h, tcr, cd45 = load_frame(frames_dir, fidx, grid_size, n_tcr, n_cd45)

        # Convert to um for scatter
        tcr_um = tcr / 1000.0
        cd45_um = cd45 / 1000.0
        tcr_scat.set_offsets(tcr_um)
        cd45_scat.set_offsets(cd45_um)

        # Depletion annulus
        tcr_r = np.sqrt(np.sum((tcr_um - c_um) ** 2, axis=1))
        cd45_r = np.sqrt(np.sum((cd45_um - c_um) ** 2, axis=1))
        inner_r = np.percentile(tcr_r, 75)
        outer_r = np.percentile(cd45_r, 25)
        band = max(0.001, outer_r - inner_r)
        depl_annulus.set_center((c_um, c_um))
        depl_annulus.set_radii(outer_r)
        depl_annulus.set_width(band)

        # Height
        im.set_data(h.T)

        # Title with physical time and depletion
        sim_step = fidx * dump_interval
        t_phys = sim_step * dt if dt > 0 else 0
        depl_w = _compute_depletion_width(tcr, cd45, patch_nm)
        t_str = f"t = {t_phys:.3f} s" if t_phys < 1 else f"t = {t_phys:.2f} s"
        title_text.set_text(f"{t_str}   |   depletion = {depl_w:.0f} nm")

        # Progress
        frac = frame_idx / max(1, len(steps) - 1)
        progress_line.set_data([0, frac], [0.5, 0.5])

        return tcr_scat, cd45_scat, im, title_text, progress_line

    ani = animation.FuncAnimation(fig, update, frames=len(steps),
                                  interval=1000 // args.fps, blit=False)

    output = args.output
    print(f"Saving to {output} ...")
    if output.endswith(".gif"):
        ani.save(output, writer="pillow", fps=args.fps, dpi=args.dpi)
    else:
        writer = animation.FFMpegWriter(
            fps=args.fps,
            extra_args=["-pix_fmt", "yuv420p"],
        )
        ani.save(output, writer=writer, dpi=args.dpi)
    print(f"Done: {output}")
    plt.close()


if __name__ == "__main__":
    main()
