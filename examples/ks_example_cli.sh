#!/usr/bin/env bash
# KS sweep example — pure CLI workflow using bayesmm commands.
#
# Usage (from repo root):
#   bash projects/tcr_signaling/examples/ks_example_cli.sh [fast|regular|extensive]
#
# Or from any directory (the script auto-detects the repo root):
#   bash /path/to/examples/ks_example_cli.sh fast
#
# Requires: bayesmm CLI installed (pip install -e . from repo root)
# Output:   projects/tcr_signaling/artifacts/ks_sweep_heatmap.png
set -euo pipefail

PROFILE="${1:-fast}"
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
echo "--- Validating spec ---"
bayesmm validate "$SPEC"
echo ""

# Step 2: Run sweep
echo "--- Running sweep ---"
time bayesmm run "$SPEC"
echo ""

# Step 3: Show results
echo "--- Run history ---"
bayesmm runs list
echo ""

# Step 4: Find the latest sweep CSV and plot
STORE_DIR="$PROJECT_DIR/store"
LATEST_CSV=$(find "$STORE_DIR/sweeps" -name "sweep_rows.csv" -type f 2>/dev/null | sort | tail -1)

if [ -z "$LATEST_CSV" ]; then
    echo "Error: no sweep_rows.csv found in $STORE_DIR/sweeps/" >&2
    exit 1
fi

echo "--- Plotting heatmap ---"
ARTIFACTS_DIR="$PROJECT_DIR/artifacts"
mkdir -p "$ARTIFACTS_DIR"
python "$SCRIPT_DIR/plot_sweep.py" \
    --csv "$LATEST_CSV" \
    --output "$ARTIFACTS_DIR" \
    --title "KS depletion width ($PROFILE profile, CLI)"

echo ""
echo "Done."
