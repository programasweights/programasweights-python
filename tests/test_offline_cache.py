#!/usr/bin/env python3
"""
Test that PAW functions work offline after first use.

Step 1 (warm): Call paw.function() with a slug to populate caches.
Step 2 (cold offline): In a NEW process with all network blocked,
       call paw.function() with the same slug and run inference.

Usage:
    python3 tests/test_offline_cache.py
"""

import json
import subprocess
import sys
import os

SLUG = "programasweights/email-triage"

STEP1_SCRIPT = '''
import programasweights as paw
from programasweights.cache import _slug_cache_path, get_cached_slug
import json

print("Step 1: Warming cache with network...")
fn = paw.function("{slug}")
result = fn("Server is down, need immediate help!")
print(f"  Result: {{result}}")

# Verify slug cache was saved
path = _slug_cache_path()
assert path.exists(), "slug_cache.json was not created!"
data = json.loads(path.read_text())
assert "{slug}" in data, "Slug not in cache!"
program_id = data["{slug}"]
print(f"  Cached slug -> {{program_id}}")
print("  STEP 1 PASSED")
'''.format(slug=SLUG)

STEP2_SCRIPT = '''
import socket

# Block ALL network access before importing anything PAW-related
_real_connect = socket.socket.connect
def _blocked_connect(self, *args, **kwargs):
    raise OSError("NETWORK BLOCKED: offline test detected a network call!")
socket.socket.connect = _blocked_connect

# Also block DNS resolution
_real_getaddrinfo = socket.getaddrinfo
def _blocked_getaddrinfo(*args, **kwargs):
    raise OSError("DNS BLOCKED: offline test detected a DNS lookup!")
socket.getaddrinfo = _blocked_getaddrinfo

import programasweights as paw

print("Step 2: Cold start with ALL network blocked...")
fn = paw.function("{slug}")
result = fn("Meeting notes from yesterday - please review")
print(f"  Result: {{result}}")
print("  STEP 2 PASSED: Function works fully offline!")
'''.format(slug=SLUG)


def main():
    python = sys.executable

    print("=" * 60)
    print("PAW Offline Cache Test")
    print("=" * 60)
    print()

    # Step 1: warm cache
    r = subprocess.run([python, "-c", STEP1_SCRIPT], capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(f"STEP 1 FAILED:\n{r.stderr}")
        return 1

    print()

    # Step 2: cold offline (new process, network blocked)
    r = subprocess.run([python, "-c", STEP2_SCRIPT], capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(f"STEP 2 FAILED:\n{r.stderr}")
        return 1

    print()
    print("=" * 60)
    print("ALL PASSED: PAW functions work offline after first use.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
