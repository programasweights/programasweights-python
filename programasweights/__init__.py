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
    fn("{name: 'Alice',}")  # '{"name":"Alice"}'

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
    __version__ = "0.4.4"

from ._output import ProgressCallback, ProgressEvent, report_progress
from .cache import CachedProgram
from .client import (
    CompileCancellation,
    CompileJob,
    CompilePrecheck,
    CompileStatus,
    Program,
)
from .config import get_api_url, get_api_key, set_api_key


def compile(
    spec: str,
    compiler: str | None = None,
    name: str | None = None,
    tags: list[str] | None = None,
    public: bool = True,
    slug: str | None = None,
    ephemeral: bool = False,
) -> Program:
    """Compile a natural language specification into a neural program.

    The compilation runs on the PAW server. The resulting program can be
    downloaded and run locally via ``paw.function(program.id)`` or
    ``paw.function(program.slug)`` if a slug was provided.

    Args:
        spec: Full specification text. Include examples in the text if desired.
        compiler: Compiler model (alias or snapshot name). If omitted, the
            server chooses the current default compiler.
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
        '{"name":"Alice"}'
    """
    from .client import PAWClient
    from ._output import status
    status("Compiling...")
    client = PAWClient(api_url=get_api_url(), api_key=get_api_key())
    result = client.compile(spec, compiler=compiler, name=name, tags=tags, public=public, slug=slug, ephemeral=ephemeral)
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


def precheck_compile(
    spec: str,
    compiler: str | None = None,
) -> CompilePrecheck:
    """Check whether a compile is cached and inspect current queue state."""
    from .client import PAWClient

    client = PAWClient(api_url=get_api_url(), api_key=get_api_key())
    return client.precheck_compile(spec, compiler=compiler)


def compile_async(
    spec: str,
    compiler: str,
    name: str | None = None,
    tags: list[str] | None = None,
    public: bool = True,
    slug: str | None = None,
    ephemeral: bool = False,
) -> CompileJob:
    """Queue a finetune compile and return immediately with a job ID.

    ``compiler`` is required because the asynchronous endpoint only supports
    explicit finetune compilers.
    """
    from .client import PAWClient

    client = PAWClient(api_url=get_api_url(), api_key=get_api_key())
    return client.compile_async(
        spec,
        compiler=compiler,
        name=name,
        tags=tags,
        public=public,
        slug=slug,
        ephemeral=ephemeral,
    )


def get_compile_status(job_id: str) -> CompileStatus:
    """Get live status for an asynchronous compile job."""
    from .client import PAWClient

    client = PAWClient(api_url=get_api_url(), api_key=get_api_key())
    return client.get_compile_status(job_id)


def cancel_compile(job_id: str) -> CompileCancellation:
    """Cancel a queued asynchronous compile job."""
    from .client import PAWClient

    client = PAWClient(api_url=get_api_url(), api_key=get_api_key())
    return client.cancel_compile(job_id)


def _coerce_program_reference(program_id_or_slug) -> str:
    if hasattr(program_id_or_slug, "id"):
        program_id_or_slug = program_id_or_slug.id
    if not isinstance(program_id_or_slug, str):
        raise TypeError("program_id_or_slug must be a string or Program")
    return program_id_or_slug


def _offline_requested(offline: bool) -> bool:
    import os

    return offline or os.environ.get("PAW_OFFLINE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _resolve_program_id(
    program_id_or_slug: str,
    *,
    offline: bool,
    progress: ProgressCallback | None = None,
) -> str:
    import re

    from .cache import (
        get_cached_slug,
        is_program_id,
        save_slug_mapping,
    )

    if is_program_id(program_id_or_slug):
        report_progress(
            progress,
            {
                "stage": "resolve",
                "status": "ready",
                "program_id": program_id_or_slug,
            },
        )
        return program_id_or_slug

    is_pinned = bool(re.search(r"@v\d+$", program_id_or_slug))
    cached = get_cached_slug(program_id_or_slug)
    if (is_pinned or offline) and cached:
        report_progress(
            progress,
            {
                "stage": "resolve",
                "status": "cached",
                "program_id": cached,
            },
        )
        return cached

    if offline:
        raise RuntimeError(
            f"No cached version of '{program_id_or_slug}'. Cannot resolve offline. "
            "Run once with internet to populate the cache."
        )

    report_progress(
        progress,
        {
            "stage": "resolve",
            "status": "resolving",
        },
        f"Resolving {program_id_or_slug}...",
    )
    from .client import PAWClient

    client = PAWClient(api_url=get_api_url(), api_key=get_api_key())
    try:
        resolved_id = client.resolve_slug(program_id_or_slug)
    except Exception:
        if cached and not is_pinned:
            report_progress(
                progress,
                {
                    "stage": "resolve",
                    "status": "cached_fallback",
                    "program_id": cached,
                },
                "Warning: could not reach server, using cached version of "
                f"{program_id_or_slug}",
            )
            return cached
        raise

    if not is_program_id(resolved_id):
        raise RuntimeError(
            f"Server returned invalid program ID for '{program_id_or_slug}'."
        )
    save_slug_mapping(program_id_or_slug, resolved_id)
    report_progress(
        progress,
        {
            "stage": "resolve",
            "status": "ready",
            "program_id": resolved_id,
        },
    )
    return resolved_id


def prepare_program(
    program_id_or_slug,
    offline: bool = False,
    progress: ProgressCallback | None = None,
) -> CachedProgram:
    """Prepare a program for later local inference without loading the model.

    Resolves a slug, downloads and validates the program bundle, resolves its
    runtime manifest, and ensures the exact base-model file is present.
    """
    from . import cache

    reference = _coerce_program_reference(program_id_or_slug)
    offline = _offline_requested(offline)
    resolved_id = _resolve_program_id(
        reference,
        offline=offline,
        progress=progress,
    )

    if offline:
        if not cache.has_valid_program_assets(resolved_id):
            raise RuntimeError(
                f"Program {resolved_id} is not fully cached and cannot be "
                "prepared offline."
            )
        report_progress(
            progress,
            {
                "stage": "program",
                "status": "cached",
                "program_id": resolved_id,
                "path": str(cache.get_program_dir(resolved_id)),
            },
        )
    else:
        from .client import PAWClient

        client = PAWClient(api_url=get_api_url(), api_key=get_api_key())
        client.download_paw(resolved_id, progress=progress)

    meta = cache.load_cached_program_meta(resolved_id)
    if meta is None:
        raise RuntimeError(
            f"Program {resolved_id} has invalid or incomplete cached metadata."
        )

    api_url = None if offline else get_api_url()
    api_key = None if offline else get_api_key()
    runtime_manifest = cache.resolve_runtime_manifest(
        meta,
        api_url=api_url,
        api_key=api_key,
        offline=offline,
    )
    if runtime_manifest is None:
        raise RuntimeError(
            f"Program {resolved_id} has no usable cached runtime manifest."
        )
    runtime_id = str(runtime_manifest.get("runtime_id", ""))
    report_progress(
        progress,
        {
            "stage": "runtime",
            "status": "ready",
            "program_id": resolved_id,
            "runtime_id": runtime_id,
        },
    )

    if offline:
        base_model_path = cache.get_cached_base_model_path(runtime_manifest)
        if base_model_path is None:
            raise RuntimeError(
                f"Base model for runtime '{runtime_id}' is not cached; "
                "cannot prepare offline."
            )
        report_progress(
            progress,
            {
                "stage": "base_model",
                "status": "cached",
                "program_id": resolved_id,
                "runtime_id": runtime_id,
                "path": str(base_model_path),
            },
        )
    else:
        interpreter = meta.get("interpreter", "Qwen/Qwen3-0.6B")
        cache.get_base_model_path(
            interpreter,
            runtime_manifest=runtime_manifest,
            progress=progress,
        )

    metadata = cache.get_cached_program_metadata(resolved_id)
    if metadata is None or not metadata["offline_ready"]:
        raise RuntimeError(
            f"Program {resolved_id} could not be made ready for offline use."
        )
    report_progress(
        progress,
        {
            "stage": "prepare",
            "status": "ready",
            "program_id": resolved_id,
            "runtime_id": runtime_id,
            "path": metadata["program_dir"],
        },
    )
    return metadata


def is_offline_ready(program_id_or_slug) -> bool:
    """Check local runnable state without making any network calls."""
    from . import cache

    reference = _coerce_program_reference(program_id_or_slug)
    resolved_id = cache.resolve_cached_program_id(reference)
    if resolved_id is None:
        return False
    metadata = cache.get_cached_program_metadata(resolved_id)
    return bool(metadata and metadata["offline_ready"])


def list_cached_programs() -> list[CachedProgram]:
    """Return metadata for valid local program caches."""
    from .cache import list_cached_programs as _list_cached_programs

    return _list_cached_programs()


def function(
    program_id,
    n_ctx: int = 2048,
    n_gpu_layers: int | None = None,
    verbose: bool = False,
    offline: bool = False,
    *,
    interpreter: str | None = None,
):
    """Load a compiled program, or explicitly load a bare base interpreter.

    Downloads the .paw bundle and base model GGUF on first use.
    Subsequent calls use the local cache.

    Args:
        program_id: Program ID (str), slug (``da03/my-program``), pinned version
            (``da03/my-program@v3``), or a ``Program`` object from compile().
        n_ctx: Context window size for llama.cpp.
        n_gpu_layers: GPU layers (-1 = all GPU, 0 = CPU only). Defaults to -1
            (auto-uses Metal/CUDA if available, safe fallback to CPU).
            Set ``PAW_GPU_LAYERS=0`` env var to force CPU-only.
        verbose: Print llama.cpp debug output.
        offline: Skip server check for slug resolution and use local cache only.
            Also set via ``PAW_OFFLINE=1`` env var.
        interpreter: Advanced adapter-free mode. This is only valid when
            ``program_id`` is explicitly ``None``. Initially supported values
            are ``"Qwen/Qwen3-0.6B"`` and ``"gpt2"``.

    Returns:
        A callable ``PawFunction`` that takes an input string and returns output.

    Example:
        >>> fn = paw.function("email-triage")
        >>> fn("Urgent: the server is down!")
        'immediate'

        >>> fn = paw.function("da03/my-program@v2")  # pinned version

        >>> base = paw.function(None, interpreter="gpt2")
    """
    import os
    from . import cache

    offline = _offline_requested(offline)
    if n_gpu_layers is None:
        n_gpu_layers = int(os.environ.get("PAW_GPU_LAYERS", "-1"))

    if program_id is None:
        if interpreter is None:
            raise ValueError(
                "program_id=None requires an explicit interpreter."
            )
        cache.get_base_runtime_manifest(interpreter)
        from .runtime_llamacpp import PawFunction

        return PawFunction.from_base(
            interpreter,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            offline=offline,
        )

    program_reference = _coerce_program_reference(program_id)
    if program_reference == "":
        raise ValueError(
            "program_id cannot be an empty string; pass explicit None with "
            "interpreter=... for adapter-free base mode."
        )
    if interpreter is not None:
        raise ValueError(
            "interpreter cannot be combined with a compiled program; pass "
            "program_id=None to request adapter-free base mode."
        )

    from .runtime_llamacpp import PawFunction

    resolved_id = _resolve_program_id(program_reference, offline=offline)
    if offline and not cache.has_valid_program_assets(resolved_id):
        raise RuntimeError(
            f"Program {resolved_id} is not fully cached; offline mode "
            "prohibits program downloads."
        )
    if not cache.has_valid_program_assets(resolved_id):
        from .client import PAWClient

        client = PAWClient(api_url=get_api_url(), api_key=get_api_key())
        client.download_paw(resolved_id)
    if not cache.has_valid_program_assets(resolved_id):
        raise RuntimeError(
            f"Program {resolved_id} is missing valid compiled assets."
        )

    program_dir = cache.get_program_dir(resolved_id)
    return PawFunction(
        program_dir,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
        api_url=None if offline else get_api_url(),
        api_key=None if offline else get_api_key(),
        offline=offline,
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
    compiler: str | None = None,
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
        compiler: Compiler model name. If omitted, the server chooses the
            current default compiler.
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


def list_compilers() -> list[dict]:
    """List available compilers from the server."""
    from .client import PAWClient
    client = PAWClient(api_url=get_api_url(), api_key=get_api_key())
    return client.list_compilers()


__all__ = [
    "CachedProgram",
    "CompileCancellation",
    "CompileJob",
    "CompilePrecheck",
    "CompileStatus",
    "Program",
    "ProgressCallback",
    "ProgressEvent",
    "cancel_compile",
    "compile",
    "compile_and_load",
    "compile_async",
    "function",
    "get_compile_status",
    "list_compilers",
    "list_cached_programs",
    "list_programs",
    "list_versions",
    "login",
    "is_offline_ready",
    "precheck_compile",
    "prepare_program",
    "get_api_url",
    "get_api_key",
    "set_api_key",
    "__version__",
]
