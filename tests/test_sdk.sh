#!/usr/bin/env bash
#
# SDK test runner: creates a fresh venv, installs from PyPI, runs all tests.
# Mimics the real user experience end-to-end.
#
# Usage:
#   bash tests/test_sdk.sh                 # uses default python3
#   bash tests/test_sdk.sh --python python3.12  # specific Python
#
set -euo pipefail

PYTHON="${PYTHON:-python3}"
EXTRA_INDEX="https://pypi.programasweights.com/simple/"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

VENV_DIR=$(mktemp -d /tmp/paw-sdk-test-XXXXXX)
echo "============================================"
echo "PAW SDK Test Suite"
echo "============================================"
echo "Python:    $($PYTHON --version 2>&1)"
echo "Venv:      $VENV_DIR"
echo "Index:     $EXTRA_INDEX"
echo ""

# Step 1: Create fresh venv
echo ">>> Creating fresh venv..."
$PYTHON -m venv "$VENV_DIR"
PIP="$VENV_DIR/bin/pip"
PY="$VENV_DIR/bin/python"

# Step 2: Install from PyPI (timed)
echo ">>> Installing programasweights from PyPI..."
START=$(date +%s)
$PIP install --no-cache-dir --upgrade programasweights --extra-index-url "$EXTRA_INDEX" -q 2>&1
END=$(date +%s)
INSTALL_TIME=$((END - START))
echo "    Install time: ${INSTALL_TIME}s"

# Verify
$PY -c "import programasweights as paw; print(f'    Version: {paw.__version__}')"

# Step 3: Install pytest
$PIP install pytest -q 2>&1

# Step 4: Copy test file
TEST_FILE="$SCRIPT_DIR/test_sdk.py"
if [ ! -f "$TEST_FILE" ]; then
  echo "ERROR: test_sdk.py not found at $TEST_FILE"
  exit 1
fi
cp "$TEST_FILE" "$VENV_DIR/test_sdk.py"

# Step 5: Run tests
echo ""
echo ">>> Running tests..."
if [ -n "$PAW_API_KEY" ]; then
  echo "    PAW_API_KEY: set (auth tests enabled)"
else
  echo "    PAW_API_KEY: not set (auth tests will be skipped)"
fi
echo ""
PAW_API_KEY="${PAW_API_KEY:-}" $PY -m pytest "$VENV_DIR/test_sdk.py" -v --tb=short 2>&1
EXIT_CODE=$?

# Step 6: Cleanup
echo ""
echo ">>> Cleaning up..."
rm -rf "$VENV_DIR"

echo ""
if [ $EXIT_CODE -eq 0 ]; then
  echo "============================================"
  echo "ALL TESTS PASSED (install: ${INSTALL_TIME}s)"
  echo "============================================"
else
  echo "============================================"
  echo "TESTS FAILED (exit code: $EXIT_CODE)"
  echo "============================================"
fi

exit $EXIT_CODE
