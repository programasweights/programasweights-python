"""
ProgramAsWeights (PAW): Compile natural language specs into tiny neural
functions that run locally.

Quick start:
    import programasweights as paw

    # Use a pre-compiled function (downloads once, runs locally forever)
    fn = paw.function("email-triage")
    fn("Urgent: server is down!")  # "immediate"

    # Compile your own from a description
    program = paw.compile("Fix malformed JSON: repair missing quotes and trailing commas")
    fn = paw.function(program.id)
    fn("{name: 'Alice',}")  # '{"name": "Alice"}'

API reference:
    paw.compile(spec)          Compile a spec on the server, returns Program
    paw.function(program_id)   Load a compiled program for local inference
    paw.login()                Save API key for higher rate limits
    paw.get_api_url()          Server URL (default: https://programasweights.com)
    paw.get_api_key()          API key (set via login() or PAW_API_KEY env var)
"""
from __future__ import annotations

try:
    from importlib.metadata import version as _meta_version
    __version__ = _meta_version("programasweights")
except Exception:
    __version__ = "0.3.0"

from .config import get_api_url, get_api_key, set_api_key


def compile(
    spec: str,
    compiler: str = "paw-4b-qwen3-0.6b",
    name: str | None = None,
    tags: list[str] | None = None,
    public: bool = True,
    slug: str | None = None,
):
    """Compile a natural language specification into a neural program.

    The compilation runs on the PAW server. The resulting program can be
    downloaded and run locally via ``paw.function(program.id)`` or
    ``paw.function(program.slug)`` if a slug was provided.

    Args:
        spec: Full specification text. Include examples in the text if desired.
        compiler: Compiler model (alias or snapshot name).
        name: Human-readable program name (display title for the hub).
        tags: Tags for hub discovery.
        public: Whether to list on the public hub.
        slug: URL-safe handle (e.g. 'message-classifier'). Creates a
            ``username/slug`` alias. Requires authentication.

    Returns:
        A ``Program`` object with ``id``, ``slug``, ``status``, and ``timings``.

    Example:
        >>> program = paw.compile(
        ...     "Fix malformed JSON: repair missing quotes and trailing commas",
        ...     slug="json-fixer"
        ... )
        >>> fn = paw.function(program.slug)  # or paw.function(program.id)
        >>> fn("{name: 'Alice',}")
        '{"name": "Alice"}'
    """
    from .client import PAWClient
    from ._output import status
    status("Compiling...")
    client = PAWClient(api_url=get_api_url(), api_key=get_api_key())
    result = client.compile(spec, compiler=compiler, name=name, tags=tags, public=public, slug=slug)
    label = f"{result.id}"
    if result.slug:
        ver_str = f" v{result.version}" if result.version and result.version > 1 else ""
        action_str = ""
        if result.version_action == "no_change":
            action_str = " (no changes)"
        elif result.version_action == "promoted":
            action_str = " set as main"
        label = f"{result.slug}{ver_str}{action_str}"
    status(f"Compiled: {label}")
    return result


def function(
    program_id,
    n_ctx: int = 2048,
    n_gpu_layers: int | None = None,
    verbose: bool = False,
    offline: bool = False,
):
    """Load a compiled program for local inference via llama.cpp.

    Downloads the .paw bundle and base model GGUF on first use.
    Subsequent calls use the local cache.

    Args:
        program_id: Program ID (str), slug (``da03/my-program``), pinned version
            (``da03/my-program@v3``), or a ``Program`` object from compile().
        n_ctx: Context window size for llama.cpp.
        n_gpu_layers: GPU layers (-1 = all, 0 = CPU only). Defaults to CPU.
            Set ``PAW_GPU_LAYERS`` env var or pass explicitly for GPU acceleration.
        verbose: Print llama.cpp debug output.
        offline: Skip server check for slug resolution and use local cache only.
            Also set via ``PAW_OFFLINE=1`` env var.

    Returns:
        A callable ``PawFunction`` that takes an input string and returns output.

    Example:
        >>> fn = paw.function("email-triage")
        >>> fn("Urgent: the server is down!")
        'immediate'

        >>> fn = paw.function("da03/my-program@v2")  # pinned version
    """
    if hasattr(program_id, 'id'):
        program_id = program_id.slug or program_id.id
    import os
    import re
    from .cache import is_program_cached, get_program_dir, get_cached_slug, save_slug_mapping
    from .runtime_llamacpp import PawFunction

    if n_gpu_layers is None:
        n_gpu_layers = int(os.environ.get("PAW_GPU_LAYERS", "0"))

    from ._output import status

    resolved_id = program_id
    if not re.fullmatch(r"[a-f0-9]{16,64}", program_id):
        is_pinned = bool(re.search(r"@v\d+$", program_id))
        use_offline = offline or os.environ.get("PAW_OFFLINE", "").strip() in ("1", "true", "yes")

        if is_pinned:
            cached = get_cached_slug(program_id)
            if cached:
                resolved_id = cached
            else:
                status(f"Resolving {program_id}...")
                from .client import PAWClient
                client = PAWClient(api_url=get_api_url(), api_key=get_api_key())
                resolved_id = client.resolve_slug(program_id)
                save_slug_mapping(program_id, resolved_id)
        elif use_offline:
            cached = get_cached_slug(program_id)
            if cached:
                resolved_id = cached
            else:
                raise RuntimeError(
                    f"No cached version of '{program_id}'. Cannot resolve offline. "
                    "Run once with internet to populate the cache."
                )
        else:
            from .client import PAWClient
            client = PAWClient(api_url=get_api_url(), api_key=get_api_key())
            try:
                status(f"Resolving {program_id}...")
                resolved_id = client.resolve_slug(program_id)
                save_slug_mapping(program_id, resolved_id)
            except Exception:
                cached = get_cached_slug(program_id)
                if cached:
                    status(f"Warning: could not reach server, using cached version of {program_id}")
                    resolved_id = cached
                else:
                    raise

    if not is_program_cached(resolved_id):
        from .client import PAWClient
        client = PAWClient(api_url=get_api_url(), api_key=get_api_key())
        client.download_paw(resolved_id)

    program_dir = get_program_dir(resolved_id)
    return PawFunction(
        program_dir, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=verbose,
    )


def login(key: str | None = None):
    """Store an API key for authenticating with the PAW server.

    If no key is provided, opens the Settings page in a browser
    and prompts interactively for the key.

    Generate your API key at https://programasweights.com/settings

    Args:
        key: API key string (``paw_sk_...``). If None, prompts interactively.

    Example:
        >>> paw.login()
        Generate an API key at https://programasweights.com/settings
        Paste your API key: ********
        API key saved.

        >>> paw.login("paw_sk_abc123...")
        API key saved.
    """
    if key is None:
        settings_url = get_api_url().rstrip("/") + "/settings"
        print(f"Generate an API key at {settings_url}")
        try:
            import webbrowser
            webbrowser.open(settings_url)
        except Exception:
            pass

        import getpass
        key = getpass.getpass("Paste your API key: ").strip()

    if not key:
        print("No key provided. Aborted.")
        return

    if not key.startswith("paw_sk_"):
        print("Warning: key doesn't start with 'paw_sk_'. Saving anyway.")

    set_api_key(key)

    print("API key saved to ~/.config/programasweights/config.json")
    print("You can also set the PAW_API_KEY environment variable.")


def compile_and_load(
    spec: str,
    compiler: str = "paw-4b-qwen3-0.6b",
    n_ctx: int = 2048,
    n_gpu_layers: int | None = None,
    verbose: bool = False,
    **compile_kwargs,
):
    """Compile a spec and immediately load it for local inference.

    Convenience wrapper that combines ``paw.compile()`` and ``paw.function()``
    into a single call.

    Args:
        spec: Natural language specification.
        compiler: Compiler model name.
        n_ctx: Context window size for llama.cpp.
        n_gpu_layers: GPU layers (-1 = all, 0 = CPU only).
        verbose: Print llama.cpp debug output.
        **compile_kwargs: Additional args passed to compile (slug, public, etc.)

    Returns:
        A callable ``PawFunction``.

    Example:
        >>> fn = paw.compile_and_load("Classify sentiment as positive or negative")
        >>> fn("I love this!")
        'positive'
    """
    program = compile(spec, compiler=compiler, **compile_kwargs)
    return function(program, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=verbose)


def list_versions(slug: str) -> dict:
    """List all versions of a named program (slug).

    Args:
        slug: Slug in ``username/slug-name`` or bare ``slug-name`` format.

    Returns:
        Dict with ``slug``, ``main_version``, and ``versions`` list.

    Example:
        >>> versions = paw.list_versions("da03/bibtex-normalizer")
        >>> for v in versions["versions"]:
        ...     print(f"v{v['version']}: {v['program_id'][:12]} {'(main)' if v['is_main'] else ''}")
    """
    from .client import PAWClient
    client = PAWClient(api_url=get_api_url(), api_key=get_api_key())
    return client.list_slug_versions(slug)


def list_programs(sort: str = "recent", per_page: int = 20, page: int = 1) -> dict:
    """List your compiled programs. Requires authentication (PAW_API_KEY).

    Returns:
        Dict with ``programs`` (list), ``total``, ``page``, ``per_page``.

    Example:
        >>> programs = paw.list_programs()
        >>> for p in programs["programs"]:
        ...     print(p["id"], p["name"])
    """
    from .client import PAWClient
    client = PAWClient(api_url=get_api_url(), api_key=get_api_key())
    return client.list_programs(sort=sort, per_page=per_page, page=page)


__all__ = [
    "compile",
    "compile_and_load",
    "function",
    "list_programs",
    "list_versions",
    "login",
    "get_api_url",
    "get_api_key",
    "set_api_key",
    "__version__",
]
