#!/usr/bin/env python3
"""Render a movie from KS simulation frame dumps (2-panel: molecules + height)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Annulus
import numpy as np

# ── Physical constants (fallbacks when not in meta.json) ─────────────
PATCH_SIZE_NM = 2000.0
INIT_HEIGHT_NM = 70.0

# ── Paul Tol bright palette (colorblind-safe) ────────────────────────
COLOR_TCR = "#EE6677"       # rose — warm, prominent
COLOR_CD45 = "#4477AA"      # blue — cool, easily separated from rose
COLOR_PMHC = "#228833"      # green — distinct from rose TCR and blue CD45
COLOR_DEPLETION = "#CCBB44"  # muted gold — annotation (partial separation)
COLOR_DEPL_GOOD = "#228833"  # green — well separated (overlap < 0.1)
COLOR_DEPL_POOR = "#CC3311"  # red — poor separation / overlap (overlap >= 0.4)


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


def _compute_depletion_metrics(tcr_pos, cd45_pos, patch_size):
    """Compute multiple depletion metrics from molecule positions (nm)."""
    center = patch_size / 2.0
    tcr_r = np.sqrt(np.sum((tcr_pos - center) ** 2, axis=1))
    cd45_r = np.sqrt(np.sum((cd45_pos - center) ** 2, axis=1))

    median_diff = max(0.0, float(np.median(cd45_r) - np.median(tcr_r)))
    tcr_p75 = float(np.percentile(tcr_r, 75))
    cd45_p25 = float(np.percentile(cd45_r, 25))
    pct_gap = cd45_p25 - tcr_p75

    # Overlap coefficient via histogram.
    r_max = patch_size * 0.7072
    bins = np.linspace(0, r_max, 101)
    h_tcr, _ = np.histogram(tcr_r, bins=bins, density=True)
    h_cd45, _ = np.histogram(cd45_r, bins=bins, density=True)
    bin_w = bins[1] - bins[0]
    overlap = float(np.sum(np.minimum(h_tcr, h_cd45)) * bin_w)

    # Frontier nearest-neighbor gap.
    ft_mask = tcr_r > tcr_p75
    fc_mask = cd45_r < cd45_p25
    frontier_nn = 0.0
    if np.any(ft_mask) and np.any(fc_mask):
        ft_pos = tcr_pos[ft_mask]
        fc_pos = cd45_pos[fc_mask]
        half = patch_size / 2.0
        nn_dists = []
        for t in ft_pos:
            dx = t[0] - fc_pos[:, 0]
            dy = t[1] - fc_pos[:, 1]
            dx = np.where(dx > half, dx - patch_size, np.where(dx < -half, dx + patch_size, dx))
            dy = np.where(dy > half, dy - patch_size, np.where(dy < -half, dy + patch_size, dy))
            nn_dists.append(np.min(np.sqrt(dx * dx + dy * dy)))
        nn_arr = np.array(nn_dists)
        lo = len(nn_arr) // 10
        hi = len(nn_arr) - 1 - len(nn_arr) // 10
        if lo > hi:
            lo, hi = 0, len(nn_arr) - 1
        frontier_nn = float(np.median(np.sort(nn_arr)[lo:hi + 1]))

    return {
        "median_diff_nm": median_diff,
        "pct_gap_nm": pct_gap,
        "overlap": overlap,
        "frontier_nn_nm": frontier_nn,
        "tcr_p75_um": tcr_p75 / 1000.0,
        "cd45_p25_um": cd45_p25 / 1000.0,
    }


def _compute_cross_nn_p10(tcr_pos, cd45_pos, pmhc_pos, patch_size,
                           bind_threshold=3.0):
    """Compute P10 cross-NN distances for bound TCRs.

    Returns (bound_mask, tcr_cd45_p10_nm, cd45_tcr_p10_nm).
    bound_mask: boolean array of shape (n_tcr,), True for bound TCRs.
    P10 values are None if no TCRs are bound.
    """
    half = patch_size / 2.0
    n_tcr = len(tcr_pos)
    n_cd45 = len(cd45_pos)

    # Find bound TCRs: within bind_threshold of any pMHC
    bound_mask = np.zeros(n_tcr, dtype=bool)
    if pmhc_pos is not None and len(pmhc_pos) > 0:
        thr2 = bind_threshold * bind_threshold
        for t in range(n_tcr):
            dx = tcr_pos[t, 0] - pmhc_pos[:, 0]
            dy = tcr_pos[t, 1] - pmhc_pos[:, 1]
            dx = np.where(dx > half, dx - patch_size,
                          np.where(dx < -half, dx + patch_size, dx))
            dy = np.where(dy > half, dy - patch_size,
                          np.where(dy < -half, dy + patch_size, dy))
            if np.any(dx * dx + dy * dy < thr2):
                bound_mask[t] = True

    n_bound = int(np.sum(bound_mask))
    if n_bound == 0:
        return bound_mask, None, None

    bound_tcr = tcr_pos[bound_mask]

    # Bound-TCR → nearest CD45
    nn_tcr_cd45 = np.empty(n_bound)
    for i in range(n_bound):
        dx = bound_tcr[i, 0] - cd45_pos[:, 0]
        dy = bound_tcr[i, 1] - cd45_pos[:, 1]
        dx = np.where(dx > half, dx - patch_size,
                      np.where(dx < -half, dx + patch_size, dx))
        dy = np.where(dy > half, dy - patch_size,
                      np.where(dy < -half, dy + patch_size, dy))
        nn_tcr_cd45[i] = np.min(np.sqrt(dx * dx + dy * dy))
    nn_tcr_cd45.sort()
    tcr_cd45_p10 = float(nn_tcr_cd45[n_bound // 10])

    # CD45 → nearest bound-TCR
    nn_cd45_tcr = np.empty(n_cd45)
    for j in range(n_cd45):
        dx = cd45_pos[j, 0] - bound_tcr[:, 0]
        dy = cd45_pos[j, 1] - bound_tcr[:, 1]
        dx = np.where(dx > half, dx - patch_size,
                      np.where(dx < -half, dx + patch_size, dx))
        dy = np.where(dy > half, dy - patch_size,
                      np.where(dy < -half, dy + patch_size, dy))
        nn_cd45_tcr[j] = np.min(np.sqrt(dx * dx + dy * dy))
    nn_cd45_tcr.sort()
    cd45_tcr_p10 = float(nn_cd45_tcr[n_cd45 // 10])

    return bound_mask, tcr_cd45_p10, cd45_tcr_p10


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
    parser.add_argument("--show-separation", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Show P10 cross-NN separation circles for bound TCRs (default: show)")
    parser.add_argument("--bind-threshold", type=float, default=3.0,
                        help="Distance threshold (nm) for TCR-pMHC binding (default: 3.0)")
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

    rigidity = args.rigidity if args.rigidity is not None else meta.get("rigidity_kT", meta.get("rigidity_kT_nm2"))
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
    # Bound TCR highlight (filled, brighter)
    bound_scat = ax_mol.scatter([], [], c=COLOR_TCR, s=40, alpha=0.9,
                                edgecolors="white", linewidths=0.8,
                                zorder=5) if args.show_separation else None
    # P10 separation circle (drawn as a single circle at patch center for legend)
    sep_circles = []  # will be re-created each frame

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

    # Separation info text (bottom-left of molecule panel)
    sep_text = ax_mol.text(0.02, 0.02, "", transform=ax_mol.transAxes,
                           fontsize=8, va="bottom", ha="left",
                           color="0.3", fontstyle="italic",
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7,
                                     ec="0.8", lw=0.5),
                           zorder=10) if args.show_separation else None

    def update(frame_idx):
        fidx = steps[frame_idx]
        h, tcr, cd45 = load_frame(frames_dir, fidx, grid_size, n_tcr, n_cd45)

        # Convert to um for scatter
        tcr_um = tcr / 1000.0
        cd45_um = cd45 / 1000.0
        tcr_scat.set_offsets(tcr_um)
        cd45_scat.set_offsets(cd45_um)

        # Depletion metrics and annulus
        metrics = _compute_depletion_metrics(tcr, cd45, patch_nm)
        inner_r = metrics["tcr_p75_um"]
        outer_r = metrics["cd45_p25_um"]
        ovl = metrics["overlap"]

        depl_annulus.set_center((c_um, c_um))
        if outer_r > inner_r:
            # Clear gap: show depletion zone
            depl_annulus.set_radii(outer_r)
            depl_annulus.set_width(outer_r - inner_r)
            depl_annulus.set_alpha(0.12)
            depl_annulus.set_linestyle("-")
            if ovl < 0.1:
                depl_annulus.set_facecolor(COLOR_DEPL_GOOD)
                depl_annulus.set_edgecolor(COLOR_DEPL_GOOD)
            elif ovl < 0.4:
                depl_annulus.set_facecolor(COLOR_DEPLETION)
                depl_annulus.set_edgecolor(COLOR_DEPLETION)
            else:
                depl_annulus.set_facecolor(COLOR_DEPL_POOR)
                depl_annulus.set_edgecolor(COLOR_DEPL_POOR)
        else:
            # Overlap regime: show overlap zone (red, dashed)
            depl_annulus.set_radii(inner_r)
            depl_annulus.set_width(max(0.001, inner_r - outer_r))
            depl_annulus.set_facecolor(COLOR_DEPL_POOR)
            depl_annulus.set_edgecolor(COLOR_DEPL_POOR)
            depl_annulus.set_alpha(0.08)
            depl_annulus.set_linestyle("--")

        # Cross-NN P10 separation visualization
        # Remove previous circles
        for c in sep_circles:
            c.remove()
        sep_circles.clear()

        if args.show_separation and pmhc_pos is not None:
            bound_mask, tcr_cd45_p10, cd45_tcr_p10 = _compute_cross_nn_p10(
                tcr, cd45, pmhc_pos, patch_nm, args.bind_threshold)
            n_bound = int(np.sum(bound_mask))

            # Highlight bound TCRs
            if n_bound > 0:
                bound_scat.set_offsets(tcr_um[bound_mask])
            else:
                bound_scat.set_offsets(np.empty((0, 2)))

            if tcr_cd45_p10 is not None:
                # Draw P10 radius circle around each bound TCR
                r_um = tcr_cd45_p10 / 1000.0
                for bi in np.where(bound_mask)[0]:
                    circ = plt.Circle(
                        (tcr_um[bi, 0], tcr_um[bi, 1]), r_um,
                        fill=False, edgecolor=COLOR_CD45, linewidth=0.6,
                        alpha=0.4, linestyle=":", zorder=4)
                    ax_mol.add_patch(circ)
                    sep_circles.append(circ)

                sep_text.set_text(
                    f"bound: {n_bound}/{n_tcr}  |  "
                    f"TCR\u2192CD45 P10: {tcr_cd45_p10:.0f} nm  |  "
                    f"CD45\u2192TCR P10: {cd45_tcr_p10:.0f} nm")
            else:
                sep_text.set_text(f"bound: 0/{n_tcr}")

        # Height
        im.set_data(h.T)

        # Title: frontier NN gap + overlap
        sim_step = fidx * dump_interval
        t_phys = sim_step * dt if dt > 0 else 0
        t_str = f"t = {t_phys:.3f} s" if t_phys < 1 else f"t = {t_phys:.2f} s"
        fnn = metrics["frontier_nn_nm"]
        gap = metrics["pct_gap_nm"]
        if gap > 0:
            title_text.set_text(
                f"{t_str}   |   front gap {fnn:.0f} nm  (overlap {ovl:.2f})")
        else:
            title_text.set_text(
                f"{t_str}   |   mixed (overlap {ovl:.2f}, gap {gap:.0f} nm)")

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
