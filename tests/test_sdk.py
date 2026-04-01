"""
Comprehensive SDK test suite for ProgramAsWeights.

Tests every documented example from the website, AGENTS.md, and docs.
Designed to run in a fresh venv with `pip install programasweights`
from PyPI -- no local repo needed.

Auth tests require PAW_API_KEY env var. They are skipped if not set.

Usage:
    pytest test_sdk.py -v
    # or via the test runner:
    bash test_sdk.sh
"""

import json
import os
import re
import socket
import subprocess
import sys

import pytest

import programasweights as paw

KNOWN_HASH = "94f4ca7b5b7973f5b407"
API_KEY = os.environ.get("PAW_API_KEY")
needs_auth = pytest.mark.skipif(not API_KEY, reason="PAW_API_KEY not set")


# ── Phase 1: Installation and import ──


class TestInstallAndImport:
    def test_import(self):
        assert hasattr(paw, "compile")
        assert hasattr(paw, "function")
        assert hasattr(paw, "login")

    def test_version(self):
        assert re.match(r"\d+\.\d+\.\d+", paw.__version__)

    def test_cli_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "programasweights.cli", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "paw" in result.stdout.lower() or "usage" in result.stdout.lower()


# ── Phase 2: Load pre-compiled programs ──


@pytest.fixture(scope="module")
def email_triage_fn():
    return paw.function("email-triage")


@pytest.fixture(scope="module")
def json_fixer_fn():
    return paw.function("json-fixer")


class TestLoadPrograms:
    def test_function_official_short_name(self, email_triage_fn):
        assert email_triage_fn is not None

    def test_function_official_full_slug(self):
        fn = paw.function("programasweights/email-triage")
        assert fn is not None

    def test_function_hash_id(self):
        fn = paw.function(KNOWN_HASH)
        assert fn is not None

    def test_function_user_slug(self):
        fn = paw.function("da03/verb-counter")
        assert fn is not None


# ── Phase 3: Local inference ──


class TestLocalInference:
    def test_email_triage_immediate(self, email_triage_fn):
        result = email_triage_fn("URGENT: production server is down, need immediate fix!")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_email_triage_wait(self, email_triage_fn):
        result = email_triage_fn("Department newsletter: spring picnic next Friday")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_email_triage_batch(self, email_triage_fn):
        messages = [
            "Thesis defense committee needs your signature by EOD",
            "Department newsletter: spring picnic next Friday",
            "URGENT: production database is down, need approval to restart",
            "FYI: new parking policy starts next month",
        ]
        results = [email_triage_fn(m) for m in messages]
        assert all(isinstance(r, str) and len(r) > 0 for r in results)

    def test_json_fixer(self, json_fixer_fn):
        result = json_fixer_fn("{name: 'Alice', age: 30,}")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_json_fixer_batch(self, json_fixer_fn):
        broken = [
            "{name: 'Alice', age: 30,}",
            "{'tags': ['ml', 'ai',], 'active': True}",
            '{key: "value", nested: {a: 1}}',
        ]
        results = [json_fixer_fn(b) for b in broken]
        assert all(isinstance(r, str) and len(r) > 0 for r in results)

    def test_log_triage(self):
        fn = paw.function("log-triage")
        log = (
            "[INFO] Server started on port 8080\n"
            "[DEBUG] Loading config from /etc/app.conf\n"
            "[ERROR] Connection refused: database timeout after 30s\n"
            "[INFO] Retrying connection (attempt 2/5)\n"
            "[WARN] Slow query detected: 2.3s\n"
            "[ERROR] Max retries exceeded, shutting down"
        )
        result = fn(log)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_fuzzy_search(self):
        fn = paw.function("fuzzy-search-ml")
        text = (
            "Our team uses ML models for NLP tasks. We also explored "
            "deep learning architectures and traditional statistical methods. "
            "The neural network approach outperformed SVM baselines."
        )
        result = fn(text)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_inference_kwargs(self, email_triage_fn):
        result = email_triage_fn("test input", max_tokens=64, temperature=0.0)
        assert isinstance(result, str)

    def test_inference_empty_input(self, email_triage_fn):
        result = email_triage_fn("")
        assert isinstance(result, str)

    def test_multiple_calls_same_function(self, email_triage_fn):
        results = [email_triage_fn("server is down") for _ in range(5)]
        assert all(isinstance(r, str) and len(r) > 0 for r in results)

    def test_two_functions_loaded(self, email_triage_fn, json_fixer_fn):
        r1 = email_triage_fn("urgent meeting")
        r2 = json_fixer_fn("{a: 1}")
        assert isinstance(r1, str) and len(r1) > 0
        assert isinstance(r2, str) and len(r2) > 0

    def test_function_reuse_across_calls(self, email_triage_fn):
        r1 = email_triage_fn("first call")
        r2 = email_triage_fn("second call")
        assert isinstance(r1, str) and isinstance(r2, str)


# ── Phase 4: Compile and use ──


class TestCompile:
    def test_compile_basic(self):
        program = paw.compile("Classify text as positive or negative sentiment.")
        assert program.id
        assert program.status == "ready"

    def test_compile_and_infer(self):
        program = paw.compile(
            "Fix malformed JSON: repair missing quotes and trailing commas"
        )
        assert program.id
        fn = paw.function(program.id)
        result = fn("{name: 'Alice',}")
        assert isinstance(result, str) and len(result) > 0


# ── Phase 5: CLI ──


class TestCLI:
    def test_cli_compile(self):
        result = subprocess.run(
            [sys.executable, "-m", "programasweights.cli",
             "compile", "--spec", "Classify sentiment as positive or negative.", "--json"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data.get("program_id")
        assert data.get("status") == "ready"

    def test_cli_info(self):
        result = subprocess.run(
            [sys.executable, "-m", "programasweights.cli",
             "info", KNOWN_HASH, "--json"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data.get("id") == KNOWN_HASH

    def test_cli_run(self):
        result = subprocess.run(
            [sys.executable, "-m", "programasweights.cli",
             "run", "--program", "email-triage", "--input", "server is down", "--json"],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data.get("output")
        assert isinstance(data["output"], str)


# ── Phase 6: Offline cold start ──


class TestOffline:
    def test_offline_cold_start(self, email_triage_fn):
        # Phase 1: ensure cache is warm (email_triage_fn fixture did this)
        _ = email_triage_fn("warm up")

        # Phase 2: new process with all network blocked
        offline_script = '''
import socket
_real_connect = socket.socket.connect
def _blocked(*a, **kw):
    raise OSError("NETWORK BLOCKED")
socket.socket.connect = _blocked

_real_gai = socket.getaddrinfo
def _blocked_gai(*a, **kw):
    raise OSError("DNS BLOCKED")
socket.getaddrinfo = _blocked_gai

import programasweights as paw
fn = paw.function("email-triage")
result = fn("test offline")
assert isinstance(result, str) and len(result) > 0
print("OFFLINE_OK")
'''
        result = subprocess.run(
            [sys.executable, "-c", offline_script],
            capture_output=True, text=True, timeout=120,
        )
        assert "OFFLINE_OK" in result.stdout, f"Offline test failed: {result.stderr}"


# ── Phase 7: Error handling ──


class TestErrors:
    def test_function_nonexistent_slug(self):
        with pytest.raises(Exception):
            paw.function("nonexistent/fake-program-xyz-999")

    def test_compile_spec_too_short(self):
        with pytest.raises(Exception):
            paw.compile("hi")

    def test_cli_info_nonexistent(self):
        result = subprocess.run(
            [sys.executable, "-m", "programasweights.cli",
             "info", "aaaaaaaaaaaaaaaa", "--json"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0 or '"error"' in result.stdout or "not found" in result.stdout.lower()

    def test_cache_unknown_interpreter(self):
        from programasweights.cache import get_base_model_path
        with pytest.raises(ValueError, match="Unknown interpreter"):
            get_base_model_path("totally-fake-model")


# ── Phase 8: Authentication ──


class TestAuth:
    @needs_auth
    def test_compile_with_slug_authenticated(self):
        import time
        slug = f"test-sdk-{int(time.time()) % 100000}"
        old_key = paw.api_key
        try:
            paw.api_key = API_KEY
            program = paw.compile(
                "Classify text as positive or negative sentiment.",
                slug=slug,
            )
            assert program.id
            assert program.slug
            assert slug in program.slug
        finally:
            paw.api_key = old_key

    @needs_auth
    def test_compile_with_slug_bad_key(self):
        old_key = paw.api_key
        try:
            paw.api_key = "paw_sk_invalid_key_12345"
            with pytest.raises(Exception):
                paw.compile(
                    "Classify text as positive or negative sentiment.",
                    slug="should-fail-auth",
                )
        finally:
            paw.api_key = old_key

    @needs_auth
    def test_cli_compile_with_api_key_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "programasweights.cli",
             "--api-key", API_KEY,
             "compile", "--spec", "Count words in the input text.", "--json"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data.get("program_id")


# ── Phase 9: Slug validation ──


class TestSlugValidation:
    @needs_auth
    def test_slug_too_short(self):
        old_key = paw.api_key
        try:
            paw.api_key = API_KEY
            with pytest.raises(Exception) as exc_info:
                paw.compile("Classify text.", slug="x")
            assert "422" in str(exc_info.value) or "invalid" in str(exc_info.value).lower()
        finally:
            paw.api_key = old_key

    @needs_auth
    def test_slug_invalid_characters(self):
        old_key = paw.api_key
        try:
            paw.api_key = API_KEY
            with pytest.raises(Exception):
                paw.compile("Classify text.", slug="BAD SLUG!")
        finally:
            paw.api_key = old_key

    @needs_auth
    def test_slug_with_uppercase_rejected(self):
        old_key = paw.api_key
        try:
            paw.api_key = API_KEY
            with pytest.raises(Exception):
                paw.compile("Classify text.", slug="MyProgram")
        finally:
            paw.api_key = old_key


# ── Phase 10: Privacy ──


class TestPrivacy:
    @needs_auth
    def test_compile_private_program(self):
        import httpx
        old_key = paw.api_key
        try:
            paw.api_key = API_KEY
            program = paw.compile("Private test program for counting vowels.", public=False)
            assert program.id

            resp = httpx.get(
                f"{paw.api_url}/api/v1/programs/{program.id}",
                timeout=10.0,
            )
            assert resp.status_code == 404, "Private program should return 404 to unauthenticated users"
        finally:
            paw.api_key = old_key

    @needs_auth
    def test_compile_public_default(self):
        import httpx
        old_key = paw.api_key
        try:
            paw.api_key = API_KEY
            program = paw.compile("Public test program for counting words.")
            assert program.id

            resp = httpx.get(
                f"{paw.api_url}/api/v1/programs/{program.id}",
                timeout=10.0,
            )
            assert resp.status_code == 200, "Public program should be accessible"
        finally:
            paw.api_key = old_key


# ── Phase 11: CLI rename ──


class TestCLIRename:
    @needs_auth
    def test_cli_rename(self):
        import time
        old_key = paw.api_key
        try:
            paw.api_key = API_KEY
            program = paw.compile("Test rename program for validation.")
            slug = f"rename-test-{int(time.time()) % 100000}"

            result = subprocess.run(
                [sys.executable, "-m", "programasweights.cli",
                 "--api-key", API_KEY,
                 "rename", program.id, slug, "--json"],
                capture_output=True, text=True,
            )
            assert result.returncode == 0
            data = json.loads(result.stdout)
            assert data.get("slug")
            assert slug in data["slug"]
        finally:
            paw.api_key = old_key

    @needs_auth
    def test_cli_compile_private(self):
        result = subprocess.run(
            [sys.executable, "-m", "programasweights.cli",
             "--api-key", API_KEY,
             "compile", "--spec", "Private CLI test.", "--private", "--json"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data.get("program_id")


# ── Phase 12: New convenience methods ──


class TestCompileAndLoad:
    def test_compile_and_load_returns_callable(self):
        fn = paw.compile_and_load("Classify text as positive or negative sentiment.")
        assert callable(fn)
        result = fn("I love this product!")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_compile_and_load_with_compiler(self):
        fn = paw.compile_and_load(
            "Count the vowels in the input. Return just the number.",
            compiler="paw-4b-qwen3-0.6b",
        )
        result = fn("hello")
        assert isinstance(result, str)


class TestFunctionAcceptsProgram:
    def test_function_with_program_object(self):
        program = paw.compile("Classify text as positive or negative sentiment.")
        fn = paw.function(program)
        assert callable(fn)
        result = fn("Great product!")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_function_with_string_still_works(self):
        fn = paw.function("email-triage")
        assert callable(fn)


class TestListPrograms:
    @needs_auth
    def test_list_programs_returns_dict(self):
        old_key = paw.api_key
        try:
            paw.api_key = API_KEY
            result = paw.list_programs()
            assert isinstance(result, dict)
            assert "programs" in result
            assert "total" in result
            assert isinstance(result["programs"], list)
        finally:
            paw.api_key = old_key

    @needs_auth
    def test_list_programs_has_entries(self):
        old_key = paw.api_key
        try:
            paw.api_key = API_KEY
            result = paw.list_programs(per_page=5)
            assert len(result["programs"]) <= 5
            if result["programs"]:
                p = result["programs"][0]
                assert "id" in p
                assert "spec" in p
        finally:
            paw.api_key = old_key


class TestStderrSuppression:
    def test_no_stderr_by_default(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import programasweights as paw; fn = paw.function('email-triage'); fn('test')"],
            capture_output=True, text=True, timeout=120,
        )
        assert "n_ctx_seq" not in result.stderr
        assert "llama_context" not in result.stderr

    def test_stderr_with_verbose(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import programasweights as paw; fn = paw.function('email-triage', verbose=True); fn('test')"],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0
