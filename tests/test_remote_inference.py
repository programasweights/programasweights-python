import builtins
import json
import socket

import httpx
import pytest

import programasweights as paw


def test_remote_inference_without_local_models(monkeypatch, tmp_path):
    program_id = "a" * 20
    cache_dir = tmp_path / "cache"
    requests, clients = [], []
    real_client, real_import = httpx.Client, builtins.__import__
    for name, value in {
        "PAW_API_URL": "https://remote.test/prefix",
        "PAW_API_KEY": "test-key",
        "PAW_OFFLINE": "0",
        "PAW_GPU_LAYERS": "invalid-local-setting",
        "PAW_CACHE_DIR": str(cache_dir),
    }.items():
        monkeypatch.setenv(name, value)

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        names = (name, *(fromlist or ()))
        if any("llama_cpp" in item or "runtime_llamacpp" in item for item in names):
            pytest.fail("Remote inference attempted to import a local runtime")
        return real_import(name, globals, locals, fromlist, level)

    def block_network(*args, **kwargs):
        pytest.fail("Unexpected real network access")

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"output": "result"})

    def make_client(**kwargs):
        client = real_client(transport=httpx.MockTransport(handle), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(socket, "create_connection", block_network)
    monkeypatch.setattr(httpx, "Client", make_client)
    with paw.function(program_id, remote=True) as fn:
        assert fn("one") == "result"
        assert fn("two") == "result"

    assert [json.loads(request.content) for request in requests] == [
        {"program_id": program_id, "input": "one"},
        {"program_id": program_id, "input": "two"},
    ]
    for request in requests:
        assert request.method == "POST"
        assert str(request.url) == "https://remote.test/prefix/api/v1/infer"
        assert request.headers["X-API-Key"] == "test-key"
    assert len(clients) == 1
    assert clients[0].is_closed
    assert not cache_dir.exists()


@pytest.mark.parametrize("entry_point", ["function", "compile_and_load"])
@pytest.mark.parametrize("options,offline_env", [
    ({}, "1"),
    ({"n_ctx": 1024}, "0"),
    ({"n_gpu_layers": 0}, "0"),
    ({"verbose": True}, "0"),
])
def test_remote_rejects_options_before_work(
    monkeypatch, entry_point, options, offline_env,
):
    monkeypatch.setenv("PAW_OFFLINE", offline_env)

    def unexpected_work(*args, **kwargs):
        pytest.fail("Invalid remote options triggered compilation or HTTP setup")

    monkeypatch.setattr(paw, "compile", unexpected_work)
    monkeypatch.setattr(httpx, "Client", unexpected_work)
    argument = "a" * 20 if entry_point == "function" else "Classify sentiment."
    message = "offline mode" if offline_env == "1" else "Local runtime options"
    with pytest.raises(ValueError, match=message):
        getattr(paw, entry_point)(argument, remote=True, **options)


@pytest.mark.parametrize("reference", [
    "email-triage", "owner/slug", "owner/slug@v3",
])
def test_remote_resolves_slugs_once(monkeypatch, reference):
    program_id = "a" * 20
    requests = []
    monkeypatch.setenv("PAW_API_URL", "https://remote.test")
    monkeypatch.setenv("PAW_API_KEY", "test-key")
    monkeypatch.setenv("PAW_OFFLINE", "0")
    monkeypatch.setattr(
        socket, "create_connection",
        lambda *args, **kwargs: pytest.fail("Unexpected real network access"),
    )

    def handle(request):
        requests.append(request)
        if request.method == "GET":
            assert request.url.path == "/api/v1/programs/resolve/" + reference
            return httpx.Response(200, json={"program_id": program_id})
        assert request.method == "POST"
        assert request.url.path == "/api/v1/infer"
        return httpx.Response(200, json={"output": "result"})

    client = httpx.Client(
        base_url="https://remote.test/", transport=httpx.MockTransport(handle),
    )
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client)
    with paw.function(reference, remote=True) as fn:
        assert fn("one") == "result"
        assert fn("two") == "result"

    assert [request.method for request in requests] == ["GET", "POST", "POST"]
    assert [json.loads(request.content)["program_id"] for request in requests[1:]] == [
        program_id, program_id,
    ]
    assert client.is_closed


@pytest.mark.parametrize("reference,status,error_code", [
    ("owner/missing", 404, "alias_not_found"),
    ("a" * 20, 503, "artifact_warming"),
])
def test_remote_errors_preserve_details_and_close_client(
    monkeypatch, reference, status, error_code,
):
    monkeypatch.setenv("PAW_API_URL", "https://remote.test")
    monkeypatch.setenv("PAW_API_KEY", "test-key")
    monkeypatch.setenv("PAW_OFFLINE", "0")
    requests = []
    payload = {
        "error": error_code, "message": "Server message", "request_id": "req-test",
    }

    def handle(request):
        requests.append(request)
        return httpx.Response(
            status,
            json={"detail": payload} if status == 404 else payload,
            headers={"Retry-After": "3"},
        )

    client = httpx.Client(
        base_url="https://remote.test/", transport=httpx.MockTransport(handle),
    )
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client)
    with pytest.raises(paw.APIError) as error:
        with paw.function(reference, remote=True) as fn:
            fn("input")

    assert len(requests) == 1
    assert error.value.response.status_code == status
    assert error.value.code == error_code
    assert error.value.message == "Server message"
    assert error.value.request_id == "req-test"
    assert error.value.response.headers["Retry-After"] == "3"
    assert client.is_closed


@pytest.mark.parametrize("generation", [
    {}, {"max_tokens": 7, "temperature": 0.0},
])
def test_compile_and_load_remote_routes_arguments(monkeypatch, generation):
    monkeypatch.setenv("PAW_API_URL", "https://remote.test")
    monkeypatch.setenv("PAW_API_KEY", "test-key")
    monkeypatch.setenv("PAW_OFFLINE", "0")
    program_id = "a" * 20
    compilations, requests = [], []

    def fake_compile(spec, **kwargs):
        compilations.append((spec, kwargs))
        return paw.Program(id=program_id, status="ready")

    def handle(request):
        assert request.method == "POST"
        assert request.url.path == "/api/v1/infer"
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"output": "result"})

    client = httpx.Client(
        base_url="https://remote.test/", transport=httpx.MockTransport(handle),
    )
    monkeypatch.setattr(paw, "compile", fake_compile)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client)
    with paw.compile_and_load(
        "Classify sentiment.", remote=True, slug="sentiment", public=False,
    ) as fn:
        assert fn("input", **generation) == "result"

    assert compilations == [(
        "Classify sentiment.",
        {"compiler": None, "slug": "sentiment", "public": False},
    )]
    assert requests == [{"program_id": program_id, "input": "input", **generation}]
    assert client.is_closed


@pytest.mark.parametrize("flags,generation,fail", [
    ([], {}, False),
    (["--max-tokens", "7", "--temperature", "0"],
     {"max_tokens": 7, "temperature": 0.0}, False),
    ([], {}, True),
])
def test_remote_cli_settings_and_cleanup(monkeypatch, capsys, flags, generation, fail):
    import sys
    from programasweights import cli

    monkeypatch.setenv("PAW_API_URL", "https://remote.test")
    monkeypatch.setenv("PAW_API_KEY", "test-key")
    monkeypatch.setenv("PAW_OFFLINE", "0")
    program_id = "a" * 20
    requests = []

    def handle(request):
        assert request.method == "POST"
        assert request.url.path == "/api/v1/infer"
        requests.append(json.loads(request.content))
        payload = (
            {"error": "artifact_warming", "message": "Preparing"}
            if fail else {"output": "result"}
        )
        return httpx.Response(503 if fail else 200, json=payload)

    client = httpx.Client(
        base_url="https://remote.test/", transport=httpx.MockTransport(handle),
    )
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client)
    monkeypatch.setattr(sys, "argv", [
        "paw", "run", "--remote", "--program", program_id,
        "--input", "input", "--json", *flags,
    ])
    if fail:
        with pytest.raises(paw.APIError, match="artifact_warming"):
            cli.main()
    else:
        assert cli.main() == 0
        assert json.loads(capsys.readouterr().out) == {
            "mode": "program", "program": program_id, "interpreter": None,
            "input": "input", "output": "result",
        }
    assert requests == [{"program_id": program_id, "input": "input", **generation}]
    assert client.is_closed


@pytest.mark.parametrize("flags,max_tokens", [
    ([], 512), (["--max-tokens", "0", "--temperature", "0"], 0),
])
def test_local_cli_preserves_defaults(monkeypatch, capsys, flags, max_tokens):
    import sys
    from programasweights import cli

    calls = []

    def fake_function(program, **kwargs):
        assert program == "a" * 20
        assert kwargs == {"verbose": False, "offline": False, "interpreter": None}

        def run(text, **generation):
            calls.append((text, generation))
            return "result"

        return run

    monkeypatch.setattr(paw, "function", fake_function)
    monkeypatch.setattr(sys, "argv", [
        "paw", "run", "--program", "a" * 20, "--input", "input", *flags,
    ])
    assert cli.main() == 0
    assert calls == [("input", {"max_tokens": max_tokens, "temperature": 0.0})]
    assert capsys.readouterr().out.strip() == "result"


@pytest.mark.parametrize("flags", [
    ["--program", "a" * 20, "--offline"],
    ["--base", "--interpreter", "gpt2"],
])
def test_remote_cli_rejects_incompatible_flags(monkeypatch, capsys, flags):
    import sys
    from programasweights import cli

    monkeypatch.setattr(
        paw, "function", lambda *a, **k: pytest.fail("Loaded an invalid request"),
    )
    monkeypatch.setattr(sys, "argv", [
        "paw", "run", "--remote", "--input", "input", *flags,
    ])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "--remote cannot be combined" in capsys.readouterr().err
