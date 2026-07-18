"""
CLI authentication plumbing tests.

These are hermetic: no network, no model load. They cover two classes of bug:

1. Regression guard for the AttributeError crash fixed in PR #1 -- `paw info`
   and `paw rename` used to read the removed module attributes
   `paw.api_url`/`paw.api_key` and crashed when no flags were given.

2. `--api-url`/`--api-key` must actually take effect on `compile`, `run`, and
   `login`. Before the 0.4.3 fix these were global flags that the high-level
   `paw.compile()`/`paw.function()`/`paw.login()` paths silently ignored
   (they only read env/config). The flag-plumbing tests below fail on that
   older behavior and pass once the flags are wired through the environment.
"""

from types import SimpleNamespace

import pytest

import programasweights as paw
from programasweights import cli, config
import programasweights.client as paw_client


class FakeProgram:
    id = "prog123"
    slug = None
    status = "ready"
    error = None
    timings = None
    version = 1
    version_action = "created"


def _install_fake_client(monkeypatch, sink):
    """Replace PAWClient with a fake that records the api_url/api_key it was
    constructed with, and returns canned data instead of hitting the network."""

    class FakeClient:
        def __init__(self, api_url=None, api_key=None):
            sink["api_url"] = api_url
            sink["api_key"] = api_key
            self._api_url = (api_url or "https://programasweights.com").rstrip("/")
            self._api_key = api_key

        def _headers(self):
            return {"Content-Type": "application/json"}

        def get_program_meta(self, program):
            return {"id": program, "spec": "canned"}

        def compile(self, *a, **k):
            return FakeProgram()

    monkeypatch.setattr(paw_client, "PAWClient", FakeClient)
    return sink


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    # Ignore any real ~/.config/programasweights/config.json and env so the
    # only source of api_url/api_key is what the command threads through.
    monkeypatch.setattr(config, "_load_config", lambda: {})
    monkeypatch.delenv("PAW_API_KEY", raising=False)
    monkeypatch.delenv("PAW_API_URL", raising=False)


# ── Regression guards: info/rename must not crash without flags ──


def test_cmd_info_no_flags_no_attributeerror(monkeypatch):
    sink = _install_fake_client(monkeypatch, {})
    args = SimpleNamespace(program="prog123", api_url=None, api_key=None, json=True)
    rc = cli.cmd_info(args)  # crashed with AttributeError before PR #1
    assert rc == 0
    assert sink["api_url"] is None and sink["api_key"] is None


def test_cmd_rename_no_flags_no_attributeerror(monkeypatch):
    sink = _install_fake_client(monkeypatch, {})

    import httpx

    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"slug": "new-slug"}

    monkeypatch.setattr(httpx, "patch", lambda *a, **k: FakeResp())
    args = SimpleNamespace(
        program="prog123", new_slug="new-slug", api_url=None, api_key=None, json=True
    )
    rc = cli.cmd_rename(args)  # crashed with AttributeError before PR #1
    assert rc == 0
    assert sink["api_url"] is None and sink["api_key"] is None


def test_cmd_info_honors_flags(monkeypatch):
    sink = _install_fake_client(monkeypatch, {})
    args = SimpleNamespace(
        program="prog123",
        api_url="http://test.local",
        api_key="paw_sk_test",
        json=True,
    )
    cli.cmd_info(args)
    assert sink["api_url"] == "http://test.local"
    assert sink["api_key"] == "paw_sk_test"


# ── Flag plumbing: compile/run/login must honor --api-url/--api-key ──


def test_apply_auth_overrides_sets_env(monkeypatch):
    cli._apply_auth_overrides(
        SimpleNamespace(api_url="http://test.local", api_key="paw_sk_test")
    )
    assert config.get_api_url() == "http://test.local"
    assert config.get_api_key() == "paw_sk_test"


def test_apply_auth_overrides_ignores_absent(monkeypatch):
    cli._apply_auth_overrides(SimpleNamespace(api_url=None, api_key=None))
    # Falls back to default url, no key.
    assert config.get_api_url() == "https://programasweights.com"
    assert config.get_api_key() is None


def test_compile_flag_plumbing(monkeypatch):
    sink = _install_fake_client(monkeypatch, {})
    args = SimpleNamespace(
        spec="Classify sentiment as positive or negative.",
        compiler=None,
        slug=None,
        private=False,
        api_url="http://test.local",
        api_key="paw_sk_test",
        json=True,
    )
    cli.cmd_compile(args)
    # The flag must reach the client through get_api_url()/get_api_key().
    assert sink["api_url"] == "http://test.local"
    assert sink["api_key"] == "paw_sk_test"


def test_run_flag_plumbing(monkeypatch):
    seen = {}

    def fake_function(program_id, **kwargs):
        seen["api_url"] = config.get_api_url()
        seen["api_key"] = config.get_api_key()
        return lambda *a, **k: "ok"

    monkeypatch.setattr(paw, "function", fake_function)
    args = SimpleNamespace(
        program="prog123",
        input="hello",
        max_tokens=512,
        temperature=0.0,
        verbose=False,
        api_url="http://test.local",
        api_key="paw_sk_test",
        json=True,
    )
    cli.cmd_run(args)
    assert seen["api_url"] == "http://test.local"
    assert seen["api_key"] == "paw_sk_test"


def test_login_flag_plumbing(monkeypatch):
    seen = {}

    def fake_login(key=None):
        seen["api_url"] = config.get_api_url()
        seen["key"] = key

    monkeypatch.setattr(paw, "login", fake_login)
    args = SimpleNamespace(
        key="paw_sk_test", api_url="http://test.local", api_key=None
    )
    cli.cmd_login(args)
    assert seen["api_url"] == "http://test.local"
    assert seen["key"] == "paw_sk_test"


def test_run_base_json_and_offline_routing(monkeypatch, capsys):
    seen = {}

    class FakeFunction:
        interpreter = "gpt2"

        def __call__(self, input_text, **kwargs):
            seen["call"] = (input_text, kwargs)
            return "base output"

    def fake_function(program_id, **kwargs):
        seen["load"] = (program_id, kwargs)
        return FakeFunction()

    monkeypatch.setattr(paw, "function", fake_function)
    args = SimpleNamespace(
        program=None,
        base=True,
        interpreter="gpt2",
        offline=True,
        input="hello",
        max_tokens=7,
        temperature=0.25,
        verbose=False,
        api_url=None,
        api_key=None,
        json=True,
    )

    assert cli.cmd_run(args) == 0
    assert seen["load"] == (
        None,
        {
            "verbose": False,
            "offline": True,
            "interpreter": "gpt2",
        },
    )
    payload = __import__("json").loads(capsys.readouterr().out)
    assert payload == {
        "mode": "base",
        "program": None,
        "interpreter": "gpt2",
        "input": "hello",
        "output": "base output",
    }


@pytest.mark.parametrize(
    "arguments",
    [
        ["run", "--program", "", "--input", "x"],
        ["run", "--base", "--input", "x"],
        [
            "run",
            "--program",
            "program-id",
            "--interpreter",
            "gpt2",
            "--input",
            "x",
        ],
        [
            "run",
            "--program",
            "program-id",
            "--base",
            "--input",
            "x",
        ],
    ],
)
def test_run_cli_rejects_invalid_mode_combinations(monkeypatch, arguments):
    monkeypatch.setattr("sys.argv", ["paw", *arguments])
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert exc_info.value.code == 2
