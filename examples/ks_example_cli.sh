#!/usr/bin/env bash
# KS sweep example — pure CLI workflow using bayesmm commands.
#
# Demonstrates the full bayesian_metamodeling pipeline:
#   1. Validate model spec
#   2. Run parameter sweep (DOE + subprocess execution + provenance)
#   3. Plot results
#   4. Optionally fit a surrogate model from sweep data
#
# Usage (from repo root):
#   bash projects/tcr_signaling/examples/ks_example_cli.sh [fast|regular|extensive]
#
# With surrogate fit (requires PyMC conda env):
#   bash projects/tcr_signaling/examples/ks_example_cli.sh fast surrogate
#
# Requires: bayesmm CLI installed (pip install -e . from repo root)
# Output:
#   projects/tcr_signaling/store/sweeps/<run_id>/  — sweep provenance
#   projects/tcr_signaling/artifacts/ks_sweep_heatmap.png
set -euo pipefail

PROFILE="${1:-fast}"
WITH_SURROGATE="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$(dirname "$PROJECT_DIR")")"

SPEC="$SCRIPT_DIR/specs/model.kinetic_segregation.${PROFILE}.json"

if [ ! -f "$SPEC" ]; then
    echo "Error: spec not found: $SPEC" >&2
    echo "Available profiles: fast, regular, extensive" >&2
    exit 1
fi

echo "=== KS CLI example (profile: $PROFILE) ==="
echo ""

# bayesmm run must execute from the repo root so that the entrypoint
# "python -m projects.tcr_signaling.models.kinetic_segregation" resolves.
cd "$REPO_DIR"

# Step 1: Validate spec
echo "--- Step 1: Validate spec ---"
bayesmm validate "$SPEC"
echo ""

# Step 2: Run sweep (DOE + subprocess execution + centralized storage)
echo "--- Step 2: Run parameter sweep ---"
time bayesmm run "$SPEC"
echo ""

# Step 3: Show run registry
echo "--- Step 3: Run history ---"
bayesmm runs list
echo ""

# Step 4: Find the latest sweep CSV and plot
STORE_DIR="$PROJECT_DIR/store"
LATEST_CSV=$(find "$STORE_DIR/sweeps" -name "sweep_rows.csv" -type f 2>/dev/null | sort | tail -1)

if [ -z "$LATEST_CSV" ]; then
    echo "Error: no sweep_rows.csv found in $STORE_DIR/sweeps/" >&2
    exit 1
fi

echo "--- Step 4: Plot heatmap ---"
ARTIFACTS_DIR="$PROJECT_DIR/artifacts"
mkdir -p "$ARTIFACTS_DIR"
python "$SCRIPT_DIR/plot_sweep.py" \
    --csv "$LATEST_CSV" \
    --output "$ARTIFACTS_DIR" \
    --title "KS depletion width ($PROFILE profile, CLI)"
echo ""

# Step 5: Optional surrogate fit
if [ "$WITH_SURROGATE" = "surrogate" ]; then
    SURROGATE_SPEC="$SCRIPT_DIR/specs/surrogate.kinetic_segregation.pymc_gp.json"
    if [ -f "$SURROGATE_SPEC" ]; then
        echo "--- Step 5: Fit surrogate from sweep data ---"
        bayesmm surrogate fit "$SURROGATE_SPEC"
        echo ""
        echo "--- Surrogate artifacts ---"
        bayesmm surrogate list
    else
        echo "Surrogate spec not found: $SURROGATE_SPEC"
    fi
fi

echo ""
echo "Output locations:"
echo "  Sweep CSV:  $LATEST_CSV"
echo "  Heatmap:    $ARTIFACTS_DIR/ks_sweep_heatmap.png"
echo "  Store:      $STORE_DIR/sweeps/"
echo ""
echo "Done."
