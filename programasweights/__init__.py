"""
ProgramAsWeights (PAW): Compile natural language specs into neural programs.

Quick start:
    import programasweights as paw

    # Compile on the server
    program = paw.compile("Classify sentiment as positive or negative")

    # Run locally via llama.cpp (downloads base model on first use)
    fn = paw.function(program.id)
    fn("I love this!")  # -> "positive"

API reference:
    paw.compile(spec)          Compile a spec on the server, returns Program
    paw.function(program_id)   Load a compiled program for local inference
    paw.login(email)           Authenticate and store API key
    paw.api_url                Server URL (default: https://programasweights.com)
    paw.api_key                API key (set via login() or PAW_API_KEY env var)
"""

__version__ = "0.1.0.dev6"

from .config import get_api_url, get_api_key, set_api_key

# Module-level settings
api_url: str = get_api_url()
api_key: str | None = get_api_key()


def compile(
    spec: str,
    compiler: str = "paw-4b-qwen3-0.6b",
    name: str | None = None,
    tags: list[str] | None = None,
    public: bool = True,
):
    """Compile a natural language specification into a neural program.

    The compilation runs on the PAW server. The resulting program can be
    downloaded and run locally via ``paw.function(program.id)``.

    Args:
        spec: Full specification text. Include examples in the text if desired.
        compiler: Compiler model (alias or snapshot name).
        name: Human-readable program name for the hub.
        tags: Tags for hub discovery.
        public: Whether to list on the public hub.

    Returns:
        A ``Program`` object with ``id``, ``status``, and ``timings``.

    Example:
        >>> program = paw.compile(
        ...     "Classify sentiment as positive or negative.\\n"
        ...     "Examples:\\n"
        ...     "Input: I love it\\nOutput: positive\\n"
        ...     "Input: I hate it\\nOutput: negative"
        ... )
        >>> print(program.id)
        '4a533a4fb0a10f219384'
    """
    from .client import PAWClient
    client = PAWClient(api_url=api_url, api_key=api_key)
    return client.compile(spec, compiler=compiler, name=name, tags=tags, public=public)


def function(
    program_id: str,
    n_ctx: int = 2048,
    n_gpu_layers: int = -1,
    verbose: bool = False,
):
    """Load a compiled program for local inference via llama.cpp.

    Downloads the .paw bundle and base model GGUF on first use.
    Subsequent calls use the local cache.

    Args:
        program_id: The program ID from ``paw.compile()``.
        n_ctx: Context window size for llama.cpp.
        n_gpu_layers: GPU layers (-1 = all, 0 = CPU only).
        verbose: Print llama.cpp debug output.

    Returns:
        A callable ``PawFunction`` that takes an input string and returns output.

    Example:
        >>> fn = paw.function("4a533a4fb0a10f219384")
        >>> fn("I love this product!")
        'positive'
    """
    import re
    from .cache import is_program_cached, get_program_dir
    from .runtime_llamacpp import PawFunction

    resolved_id = program_id
    if not re.fullmatch(r"[a-f0-9]{16,64}", program_id):
        from .client import PAWClient
        client = PAWClient(api_url=api_url, api_key=api_key)
        resolved_id = client.resolve_slug(program_id)

    if not is_program_cached(resolved_id):
        from .client import PAWClient
        client = PAWClient(api_url=api_url, api_key=api_key)
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
        settings_url = api_url.rstrip("/") + "/settings"
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

    global api_key
    api_key = key

    print("API key saved to ~/.config/programasweights/config.json")
    print("You can also set the PAW_API_KEY environment variable.")


__all__ = [
    "compile",
    "function",
    "login",
    "api_url",
    "api_key",
    "__version__",
]
