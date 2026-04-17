#!/usr/bin/env python3
"""Quick smoke test for the main API endpoints."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
from pathlib import Path

import requests


DEFAULT_API_BASE_URL = "http://localhost:8000"
ENDPOINTS = [
    ("/predict", "Full model (NLP + CV+Audio)"),
    ("/test/nlp", "NLP-only model"),
    ("/predict/cv-audio", "CV+Audio model"),
]
DEFAULT_TEST_LOCATIONS = [
    Path("examples/sample_data/test_video.mp4"),
    Path("examples/sample_data/test.mp4"),
]


def resolve_test_file(explicit_path: str | None) -> Path | None:
    candidates: list[Path] = []

    if explicit_path:
        candidates.append(Path(explicit_path))

    env_path = os.getenv("TEST_VIDEO_PATH")
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(DEFAULT_TEST_LOCATIONS)

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.exists():
            print(f"Found test file: {resolved}")
            return resolved

    return None


def guess_content_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def test_endpoint(api_base_url: str, endpoint_path: str, endpoint_name: str, test_file: Path) -> bool:
    print(f"\n{'=' * 60}")
    print(f"Testing: {endpoint_name}")
    print(f"Endpoint: POST {endpoint_path}")
    print(f"File: {test_file}")
    print(f"{'=' * 60}")

    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return False

    try:
        with test_file.open("rb") as handle:
            files = {
                "file": (test_file.name, handle, guess_content_type(test_file)),
            }
            url = f"{api_base_url}{endpoint_path}"
            print(f"Sending request to: {url}")

            response = requests.post(url, files=files, timeout=300)
            print(f"Status Code: {response.status_code}")

            if response.status_code != 200:
                print(f"Request failed with status {response.status_code}")
                print(f"Response: {response.text[:400]}")
                return False

            result = response.json()
            print("Success")
            print(json.dumps(result, indent=2, ensure_ascii=False)[:700])

            if endpoint_path == "/predict":
                return "overall" in result and "models" in result
            if endpoint_path == "/test/nlp":
                return "prediction" in result and "risk_score" in result
            if endpoint_path == "/predict/cv-audio":
                return "prediction" in result

            return True
    except requests.exceptions.ConnectionError:
        print(f"Connection error: cannot connect to {api_base_url}")
        print("Start the stack with `python run_all.py --auto-yes` or `docker compose up --build`.")
        return False
    except Exception as exc:
        print(f"Error: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the main API endpoints.")
    parser.add_argument("video", nargs="?", help="Path to a local video file")
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL, help="API base URL")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print("API ENDPOINT TEST SUITE")
    print(f"{'=' * 60}")

    try:
        response = requests.get(f"{args.api_base_url}/health", timeout=5)
        print("API is running and responding")
        print(f"Health check: {response.json()}")
    except Exception as exc:
        print(f"API is not responding: {exc}")
        print("Start the stack with `python run_all.py --auto-yes` or `docker compose up --build`.")
        return 1

    test_file = resolve_test_file(args.video)
    if test_file is None:
        print("No test video found.")
        print("Pass a path explicitly, set TEST_VIDEO_PATH, or place a file in examples/sample_data/.")
        return 1

    results: dict[str, bool] = {}
    for endpoint_path, endpoint_name in ENDPOINTS:
        results[endpoint_path] = test_endpoint(args.api_base_url, endpoint_path, endpoint_name, test_file)

    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")

    passed = sum(1 for success in results.values() if success)
    total = len(results)

    for endpoint_path, endpoint_name in ENDPOINTS:
        status = "PASS" if results[endpoint_path] else "FAIL"
        print(f"{status}: {endpoint_name} ({endpoint_path})")

    print(f"\nTotal: {passed}/{total} endpoints working")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
