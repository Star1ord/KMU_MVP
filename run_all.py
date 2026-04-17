#!/usr/bin/env python3
"""Run backend (uvicorn) and frontend (static server) together.

Usage:
  python run_all.py [--no-browser] [--auto-yes] [--python PATH_TO_PYTHON]

This script launches two subprocesses and forwards their output with prefixes.
Ctrl+C will terminate both.

The script attempts to find Python 3.10. If not found, it will use the current interpreter
and warn you. You can override with --python /path/to/python310.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser


def stream_output(pipe, prefix):
    try:
        for line in iter(pipe.readline, b""):
            try:
                print(f"[{prefix}] {line.decode().rstrip()}")
            except Exception:
                print(f"[{prefix}] {line}")
    finally:
        pipe.close()


def _version_string(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            [*command, "--version"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        return (result.stdout + result.stderr).strip()
    except Exception:
        return None


def _display_command(command: list[str]) -> str:
    return " ".join(command)


def find_python310(override=None):
    """Try to find Python 3.10 in common locations."""
    if override:
        command = [override]
        version = _version_string(command)
        if version:
            print(f"[info] Using Python (override): {override} ({version})")
            return command
        print(f"[error] Override Python failed: {override}")
        return [sys.executable]

    candidates = [
        ["py", "-3.10"],
        ["python3.10"],
        ["python3"],
        [sys.executable],
    ]

    for candidate in candidates:
        version = _version_string(candidate)
        if version and "3.10" in version:
            print(f"[info] Found Python 3.10: {_display_command(candidate)}")
            print(f"[info] Version: {version}")
            return candidate

    current = [sys.executable]
    version = _version_string(current)
    if version:
        print(f"[warning] Python 3.10 NOT found. Using current: {version}")
        print("[warning] This project is pinned for Python 3.10.")
        print("[warning] If tests fail, install Python 3.10 or pass --python /path/to/python310")

    return current


def main():
    open_browser = True
    auto_yes = False
    python_override = None
    args = sys.argv[1:]

    if "--no-browser" in args:
        open_browser = False
    if "--auto-yes" in args:
        auto_yes = True
    if "--python" in args:
        idx = args.index("--python")
        if idx + 1 < len(args):
            python_override = args[idx + 1]

    if auto_yes:
        os.environ.setdefault("AUTO_YES", "1")

    python_cmd = find_python310(override=python_override)
    print(f"[info] Running with: {_display_command(python_cmd)}\n")

    repo_root = os.getcwd()
    if os.path.basename(repo_root) == "web_ui":
        repo_root = os.path.dirname(repo_root)

    print(f"[info] Working directory: {repo_root}\n")

    backend_cmd = [*python_cmd, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
    if "--reload" in args:
        backend_cmd.insert(-2, "--reload")
    frontend_cmd = [*python_cmd, "-m", "http.server", "5500"]

    p_backend = subprocess.Popen(
        backend_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=repo_root,
    )
    t_backend = threading.Thread(target=stream_output, args=(p_backend.stdout, "backend"), daemon=True)
    t_backend.start()

    frontend_cwd = os.path.join(repo_root, "web_ui")
    p_frontend = subprocess.Popen(
        frontend_cmd,
        cwd=frontend_cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    t_frontend = threading.Thread(target=stream_output, args=(p_frontend.stdout, "frontend"), daemon=True)
    t_frontend.start()

    if open_browser:
        try:
            webbrowser.open("http://localhost:5500")
        except Exception:
            pass

    def terminate_processes():
        print("\nStopping processes...")
        try:
            p_backend.terminate()
        except Exception:
            pass
        try:
            p_frontend.terminate()
        except Exception:
            pass

    def handle_sigint(sig, frame):
        terminate_processes()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        while True:
            if p_backend.poll() is not None:
                print("[backend] Backend exited")
                break
            if p_frontend.poll() is not None:
                print("[frontend] Frontend exited")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        terminate_processes()
        sys.exit(0)


if __name__ == "__main__":
    main()
