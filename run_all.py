#!/usr/bin/env python3
"""Run backend (uvicorn) and frontend (static server) together.

Usage:
  python run_all.py [--no-browser] [--auto-yes] [--python PATH_TO_PYTHON]

This script launches two subprocesses and forwards their output with prefixes.
Ctrl+C will terminate both.

The script attempts to find Python 3.10. If not found, it will use the current interpreter
and warn you. You can override with --python /path/to/python310.
"""
import subprocess
import threading
import webbrowser
import os
import sys
import signal
import time


def stream_output(pipe, prefix):
    try:
        for line in iter(pipe.readline, b""):
            try:
                print(f"[{prefix}] {line.decode().rstrip()}")
            except Exception:
                print(f"[{prefix}] {line}")
    finally:
        pipe.close()


def find_python310(override=None):
    """Try to find Python 3.10 in common locations."""
    if override:
        try:
            result = subprocess.run(
                [override, "--version"],
                capture_output=True,
                text=True,
                timeout=2
            )
            version = result.stdout + result.stderr
            print(f"[info] Using Python (override): {override} ({version.strip()})")
            return override
        except Exception as e:
            print(f"[error] Override Python failed: {e}")
            return sys.executable

    candidates = [
        "py -3.10",
        "python3.10",
        "python3",
        sys.executable,
    ]
    
    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                text=True,
                timeout=2
            )
            version_str = result.stdout + result.stderr
            if "3.10" in version_str:
                print(f"[info] Found Python 3.10: {candidate}")
                print(f"[info] Version: {version_str.strip()}")
                return candidate
        except Exception:
            pass
    
    # Fallback: use current sys.executable and warn
    try:
        result = subprocess.run(
            [sys.executable, "--version"],
            capture_output=True,
            text=True,
            timeout=2
        )
        version = result.stdout + result.stderr
        print(f"[warning] Python 3.10 NOT found. Using current: {version.strip()}")
        print(f"[warning] This project is pinned for Python 3.10.")
        print(f"[warning] If tests fail, install Python 3.10 or pass --python /path/to/python310")
    except Exception:
        pass
    
    return sys.executable


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

    python = find_python310(override=python_override)
    print(f"[info] Running with: {python}\n")

    # Get the repo root (parent of current directory if we're in web_ui, otherwise current)
    repo_root = os.getcwd()
    if os.path.basename(repo_root) == 'web_ui':
        repo_root = os.path.dirname(repo_root)
    
    print(f"[info] Working directory: {repo_root}\n")

    # By default do not use --reload to avoid the reloader spawning subprocesses
    # that can interfere with child pipeline processes. Use --reload if explicitly requested.
    backend_cmd = [python, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
    if "--reload" in args:
        backend_cmd.insert(-2, "--reload")
    frontend_cmd = [python, "-m", "http.server", "5500"]

    # Start backend (must run from repo root for models/ and data/ paths to work)
    p_backend = subprocess.Popen(backend_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=repo_root)
    t_backend = threading.Thread(target=stream_output, args=(p_backend.stdout, "backend"), daemon=True)
    t_backend.start()

    # Start frontend (serve web_ui)
    frontend_cwd = os.path.join(repo_root, "web_ui")
    p_frontend = subprocess.Popen(frontend_cmd, cwd=frontend_cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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

    # Wait for processes (portable loop; signal.pause not available on Windows)
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
