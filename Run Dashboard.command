#!/bin/bash
# Starts the Scalable Performance Dashboard and opens it in your default
# browser. Double-click this file in Finder. Close this window or press
# Ctrl-C to stop the dashboard.

cd "$(dirname "$0")"

clear
echo "================================================================"
echo "   Scalable Performance Dashboard"
echo "================================================================"
echo

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' is not installed. Please run 'Setup.command' first."
    echo
    read -n 1 -s -r -p "Press any key to close this window..."
    echo
    exit 1
fi

if ! ls transaction_data/*.csv >/dev/null 2>&1; then
    cat <<'EOF'
ERROR: No CSV file in "transaction_data".

Please:
  1. Export your transactions from Scalable Capital as CSV.
  2. Drop the file into the "transaction_data" folder.
  3. Double-click this launcher again.

EOF
    read -n 1 -s -r -p "Press any key to close this window..."
    echo
    exit 1
fi

cat <<'EOF'
Starting the dashboard...

  * Your browser will open at http://127.0.0.1:8050 as soon as it is ready.
  * The very first start can take 5-7 minutes while live prices and
    historical data are fetched. Later starts are fast.
  * To stop the dashboard, close this Terminal window or press Ctrl-C.

------------------------------------------------------------------
EOF

# Launch a tiny helper that opens the browser the moment the server is
# accepting connections, then start the dashboard in the foreground.
( uv run python -m dashboard.open_browser >/dev/null 2>&1 ) &
uv run python run.py
