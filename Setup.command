#!/bin/bash
# One-time setup for the Scalable Performance Dashboard on macOS.
# Double-click this file in Finder.

set -e
cd "$(dirname "$0")"

clear
echo "================================================================"
echo "   Scalable Performance Dashboard  —  Setup"
echo "================================================================"
echo

# Step 0: ensure uv is installed.
if ! command -v uv >/dev/null 2>&1; then
    cat <<'EOF'
ERROR: 'uv' is not installed.

uv is the tool that manages Python + dependencies for you. It's free and
takes 30 seconds to install.

>>> Please open "How to install.html" first and follow Step 1.

EOF
    read -n 1 -s -r -p "Press any key to close this window..."
    echo
    exit 1
fi

echo "[1/3] Installing Python and Python dependencies..."
echo "      (this may take a few minutes the first time)"
echo
uv sync

echo
echo "[2/3] Installing the headless browser (for live gettex prices)..."
echo
uv run playwright install chromium

echo
echo "[3/3] Setup complete!"
echo
cat <<'EOF'
What to do next
---------------
  1. Export your transaction history from Scalable Capital (CSV).
  2. Drop the CSV into the folder named "transaction_data" (right next
     to this Setup file).
  3. Double-click "Run Dashboard.command" to start the dashboard.

EOF

read -n 1 -s -r -p "Press any key to close this window..."
echo
