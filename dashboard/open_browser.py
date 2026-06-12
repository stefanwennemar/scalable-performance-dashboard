"""Small helper used by the launcher scripts. Polls 127.0.0.1:8050 until the
dashboard's Flask process is accepting connections, then opens the default
browser. Times out after 10 minutes so it never hangs forever."""

from __future__ import annotations

import socket
import sys
import time
import webbrowser

URL = "http://127.0.0.1:8050"
HOST = "127.0.0.1"
PORT = 8050
TIMEOUT_SECONDS = 600


def is_up() -> bool:
    try:
        with socket.create_connection((HOST, PORT), timeout=1):
            return True
    except OSError:
        return False


def main() -> int:
    deadline = time.time() + TIMEOUT_SECONDS
    while time.time() < deadline:
        if is_up():
            webbrowser.open(URL)
            return 0
        time.sleep(2)
    print("Timed out waiting for the dashboard to start.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
