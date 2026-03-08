"""Generate an animation of the kinetic segregation Monte Carlo simulation.

Produces an MP4 movie showing TCR (red) and CD45 (blue) molecules on the
membrane patch as the simulation evolves. The membrane height field is shown
as a background colormap. A depletion zone ring is overlaid.

Usage::

    python -m projects.tcr_signaling.models.kinetic_segregation.animate \\
        --output ks_segregation.mp4 --n_steps 5000 --fps 15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .model import PATCH_SIZE_NM, _compute_depletion_width, simulate_ks


def render_animation(
    snapshots: list[dict],
    patch_size: float = PATCH_SIZE_NM,
    output: str = "ks_segregation.mp4",
    fps: int = 15,
    dpi: int = 120,
    pmhc_pos: np.ndarray | None = None,
) -> Path:
    """Render simulation snapshots to an MP4 animation.

    Parameters
    ----------
    snapshots : List of dicts with keys tcr_pos, cd45_pos, h, step.
    patch_size : Patch size in nm.
    output : Output file path.
    fps : Frames per second.
    dpi : Resolution.

    Returns
    -------
    Path to the saved animation file.
    """
    fig, (ax_main, ax_h) = plt.subplots(1, 2, figsize=(14, 6))
    fig.set_facecolor("white")

    center = patch_size / 2.0
    extent_um = [0, patch_size / 1000, 0, patch_size / 1000]  # convert nm to um

    # --- Left panel: molecule positions ---
    ax_main.set_xlim(0, patch_size / 1000)
    ax_main.set_ylim(0, patch_size / 1000)
    ax_main.set_aspect("equal")
    ax_main.set_xlabel("x (um)")
    ax_main.set_ylabel("y (um)")

    tcr_scatter = ax_main.scatter([], [], c="red", s=20, alpha=0.7, label="TCR", zorder=3)
    cd45_scatter = ax_main.scatter([], [], c="royalblue", s=12, alpha=0.5, label="CD45", zorder=2)
    if pmhc_pos is not None:
        pmhc_um = pmhc_pos / 1000.0
        ax_main.scatter(pmhc_um[:, 0], pmhc_um[:, 1], c="black", marker="x",
                        s=30, alpha=0.6, label="pMHC", zorder=1, linewidths=1)
    depletion_circle_inner = plt.Circle(
        (center / 1000, center / 1000), 0, fill=False, color="gold", lw=2, ls="--", zorder=4
    )
    depletion_circle_outer = plt.Circle(
        (center / 1000, center / 1000), 0, fill=False, color="gold", lw=2, ls="--", zorder=4
    )
    ax_main.add_patch(depletion_circle_inner)
    ax_main.add_patch(depletion_circle_outer)
    ax_main.legend(loc="upper right", fontsize=9)
    title_text = ax_main.set_title("Step 0")

    # --- Right panel: membrane height field ---
    h0 = snapshots[0]["h"]
    im = ax_h.imshow(
        h0.T if h0.shape[0] == h0.shape[1] else h0,
        cmap="viridis",
        origin="lower",
        extent=extent_um,
        vmin=0,
        vmax=50,
        aspect="equal",
    )
    ax_h.set_xlabel("x (um)")
    ax_h.set_ylabel("y (um)")
    ax_h.set_title("Membrane height (nm)")
    fig.colorbar(im, ax=ax_h, label="h (nm)", shrink=0.8)

    fig.tight_layout()

    def update(frame_idx):
        snap = snapshots[frame_idx]
        tcr = snap["tcr_pos"] / 1000.0  # nm -> um
        cd45 = snap["cd45_pos"] / 1000.0

        tcr_scatter.set_offsets(tcr)
        cd45_scatter.set_offsets(cd45)

        # Compute radial statistics for depletion zone visualization
        c_um = center / 1000.0
        tcr_r = np.sqrt(np.sum((tcr - c_um) ** 2, axis=1))
        cd45_r = np.sqrt(np.sum((cd45 - c_um) ** 2, axis=1))
        tcr_75 = np.percentile(tcr_r, 75)
        cd45_25 = np.percentile(cd45_r, 25)

        depletion_circle_inner.set_radius(tcr_75)
        depletion_circle_outer.set_radius(cd45_25)

        # Update membrane height
        im.set_data(snap["h"])

        depletion_w = _compute_depletion_width(snap["tcr_pos"], snap["cd45_pos"], patch_size)
        title_text.set_text(f"Step {snap['step']}  |  depletion width = {depletion_w:.0f} nm")

        return tcr_scatter, cd45_scatter, im, title_text

    anim = FuncAnimation(fig, update, frames=len(snapshots), interval=1000 // fps, blit=False)

    out_path = Path(output)
    suffix = out_path.suffix.lower()
    if suffix == ".gif":
        anim.save(str(out_path), writer="pillow", fps=fps, dpi=dpi)
    else:
        anim.save(str(out_path), writer="ffmpeg", fps=fps, dpi=dpi)
    plt.close(fig)
    print(f"Animation saved to {out_path} ({len(snapshots)} frames, {fps} fps)")
    return out_path


def _merge_params(args: argparse.Namespace, params_dict: dict) -> None:
    """Apply param file values where CLI left defaults (None)."""
    for key, val in params_dict.items():
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, val)


def main() -> int:
    parser = argparse.ArgumentParser(description="Animate kinetic segregation simulation")
    parser.add_argument("--params", type=str, default=None, help="JSON parameter file")
    parser.add_argument("--output", type=str, default="ks_segregation.gif", help="Output file")
    parser.add_argument("--time_sec", type=float, default=None)
    parser.add_argument("--rigidity_kT_nm2", type=float, default=None)
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--n_tcr", type=int, default=None)
    parser.add_argument("--n_cd45", type=int, default=None)
    parser.add_argument("--grid_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--snapshot_every", type=int, default=None, help="Record every N steps")
    parser.add_argument("--n_pmhc", type=int, default=None, help="Number of static pMHC molecules")
    parser.add_argument("--pmhc_seed", type=int, default=None, help="pMHC seed")
    parser.add_argument("--pmhc_mode", type=str, default=None, help="pMHC placement: 'uniform' or 'inner_circle'")
    parser.add_argument("--pmhc_radius", type=float, default=None, help="pMHC placement radius (nm)")
    args = parser.parse_args()

    # Load param file and merge (CLI > param file > built-in defaults)
    if args.params is not None:
        with open(args.params) as f:
            _merge_params(args, json.load(f))

    # Apply built-in defaults
    if args.time_sec is None:
        args.time_sec = 50.0
    if args.rigidity_kT_nm2 is None:
        args.rigidity_kT_nm2 = 20.0
    if args.n_steps is None:
        args.n_steps = 5000
    if args.n_tcr is None:
        args.n_tcr = 50
    if args.n_cd45 is None:
        args.n_cd45 = 100
    if args.grid_size is None:
        args.grid_size = 32
    if args.seed is None:
        args.seed = 42
    if args.snapshot_every is None:
        args.snapshot_every = 50
    if args.n_pmhc is None:
        args.n_pmhc = 0

    print(f"Running KS simulation: {args.n_steps} steps, snapshots every {args.snapshot_every}...")
    sim_kwargs = dict(
        time_sec=args.time_sec,
        rigidity_kT_nm2=args.rigidity_kT_nm2,
        n_tcr=args.n_tcr,
        n_cd45=args.n_cd45,
        grid_size=args.grid_size,
        n_steps=args.n_steps,
        seed=args.seed,
        snapshot_interval=args.snapshot_every,
        n_pmhc=args.n_pmhc,
    )
    if args.pmhc_seed is not None:
        sim_kwargs["pmhc_seed"] = args.pmhc_seed
    if args.pmhc_mode is not None:
        sim_kwargs["pmhc_mode"] = args.pmhc_mode
    if args.pmhc_radius is not None:
        sim_kwargs["pmhc_radius"] = args.pmhc_radius
    result = simulate_ks(**sim_kwargs)
    print(f"Simulation done. Depletion width = {result['depletion_width_nm']:.1f} nm")
    print(f"Rendering {len(result['snapshots'])} frames...")

    pmhc_pos = result.get("pmhc_pos", None)
    render_animation(result["snapshots"], output=args.output, fps=args.fps, pmhc_pos=pmhc_pos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
