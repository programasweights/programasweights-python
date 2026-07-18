#!/usr/bin/env python3
"""
ProgramAsWeights CLI.

Usage:
    paw compile --spec "..."       Compile a spec on the server
    paw run --program <id> --input "..."   Run a program locally
    paw rename <program> <slug>    Set or change a program's slug
    paw info <program>             Show program metadata
    paw login [key]                Save API key for authentication

All commands support --json for structured output (agent-friendly).
"""

import argparse
import json
import os
import sys


def _apply_auth_overrides(args):
    """Let the global --api-url/--api-key flags take effect on commands that go
    through the high-level ``paw.compile()``/``paw.function()``/``paw.login()``
    paths, which resolve credentials from the environment/config only."""
    if getattr(args, "api_url", None):
        os.environ["PAW_API_URL"] = args.api_url
    if getattr(args, "api_key", None):
        os.environ["PAW_API_KEY"] = args.api_key


def _non_empty_program(value):
    if not isinstance(value, str) or not value.strip():
        raise argparse.ArgumentTypeError(
            "--program must be a non-empty program ID or slug"
        )
    return value


def cmd_compile(args):
    import programasweights as paw
    _apply_auth_overrides(args)

    if not args.json:
        print(f"Compiling: {args.spec[:80]}...")

    public = not getattr(args, 'private', False)
    program = paw.compile(args.spec, compiler=args.compiler, slug=getattr(args, 'slug', None), public=public)

    if args.json:
        print(json.dumps({
            "program_id": program.id,
            "slug": program.slug,
            "status": program.status,
            "error": program.error,
            "timings": program.timings,
        }))
        return 1 if program.error else 0

    if program.error:
        print(f"Error: {program.error}")
        return 1

    print(f"Program ID: {program.id}")
    if program.slug:
        print(f"Slug: {program.slug}")
    print(f"Status: {program.status}")
    if program.timings:
        total = program.timings.get("total_ms", 0)
        print(f"Total time: {total:.0f}ms")
    ref = program.slug or program.id
    print(f"\nTo run locally:")
    print(f"  paw run --program \"{ref}\" --input \"your input here\"")
    print(f"\nOr in Python:")
    print(f"  import programasweights as paw")
    print(f"  fn = paw.function(\"{ref}\")")
    print(f"  fn(\"your input here\")")
    return 0


def cmd_run(args):
    import programasweights as paw
    _apply_auth_overrides(args)

    base_mode = bool(getattr(args, "base", False))
    program = getattr(args, "program", None)
    interpreter = getattr(args, "interpreter", None)
    offline = bool(getattr(args, "offline", False))
    if base_mode:
        if program is not None:
            raise ValueError("--base cannot be combined with --program.")
        if interpreter is None:
            raise ValueError("--base requires --interpreter.")
    else:
        if program is None:
            raise ValueError("--program is required unless --base is used.")
        if interpreter is not None:
            raise ValueError(
                "--interpreter is only valid together with --base."
            )

    fn = paw.function(
        None if base_mode else program,
        verbose=args.verbose,
        offline=offline,
        interpreter=interpreter,
    )
    result = fn(args.input, max_tokens=args.max_tokens, temperature=args.temperature)

    if args.json:
        print(
            json.dumps(
                {
                    "mode": "base" if base_mode else "program",
                    "program": program,
                    "interpreter": getattr(fn, "interpreter", interpreter),
                    "input": args.input,
                    "output": result,
                }
            )
        )
    else:
        print(result)
    return 0


def cmd_login(args):
    import programasweights as paw
    _apply_auth_overrides(args)
    paw.login(args.key)
    return 0


def cmd_rename(args):
    import httpx
    from programasweights.client import PAWClient
    client = PAWClient(api_url=args.api_url, api_key=args.api_key)

    resp = httpx.patch(
        f"{client._api_url}/api/v1/programs/{args.program}",
        json={"slug": args.new_slug},
        headers=client._headers(),
        timeout=10.0,
    )
    resp.raise_for_status()
    data = resp.json()

    if args.json:
        print(json.dumps(data))
    else:
        slug = data.get("slug")
        if slug:
            print(f"Renamed to: {slug}")
        else:
            print("Slug removed.")
    return 0


def cmd_info(args):
    from programasweights.client import PAWClient
    client = PAWClient(api_url=args.api_url, api_key=args.api_key)

    try:
        meta = client.get_program_meta(args.program)
    except Exception:
        meta = None

    if meta and meta.get("id"):
        if args.json:
            print(json.dumps(meta))
        else:
            print(f"Program: {meta.get('id')}")
            print(f"  Spec: {(meta.get('spec') or 'N/A')[:100]}")
            print(f"  Interpreter: {meta.get('interpreter', 'N/A')}")
            print(f"  Compiler: {meta.get('compiler_snapshot', 'N/A')}")
            print(f"  Aliases: {', '.join(meta.get('aliases', []))}")
            print(f"  Downloads: {meta.get('downloads', 0)}")
            if meta.get('hf_url'):
                print(f"  HF URL: {meta['hf_url']}")
        return 0

    if args.json:
        print(json.dumps({"error": "not_found", "program": args.program}))
    else:
        print(f"Program {args.program} not found.")
    return 1


def main():
    parser = argparse.ArgumentParser(
        prog="paw",
        description="ProgramAsWeights: compile and run neural programs",
    )
    parser.add_argument("--api-url", default=None, help="PAW server URL")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--json", action="store_true",
                        help="Output structured JSON (agent-friendly)")

    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("compile", help="Compile a spec on the server")
    p.add_argument("--spec", required=True, help="Natural language specification")
    p.add_argument("--compiler", default=None, help="Compiler model (omit to use the server default)")
    p.add_argument("--slug", default=None, help="URL-safe handle (e.g. 'message-classifier')")
    p.add_argument("--private", action="store_true", help="Make program private (not listed on hub)")
    p.add_argument("--json", action="store_true", help="JSON output")

    p = sub.add_parser("run", help="Run a program locally via llama.cpp")
    run_mode = p.add_mutually_exclusive_group(required=True)
    run_mode.add_argument(
        "--program",
        type=_non_empty_program,
        help="Compiled program name or ID",
    )
    run_mode.add_argument(
        "--base",
        action="store_true",
        help="Run a bare base model (advanced; requires --interpreter)",
    )
    p.add_argument(
        "--interpreter",
        choices=("Qwen/Qwen3-0.6B", "gpt2"),
        help="Base interpreter; only valid with --base",
    )
    p.add_argument("--input", required=True, help="Input text")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--offline",
        action="store_true",
        help="Require all model/program assets to already be cached",
    )
    p.add_argument("--json", action="store_true", help="JSON output")

    p = sub.add_parser("login", help="Save API key for authentication")
    p.add_argument("key", nargs="?", default=None, help="API key (paw_sk_...). Omit to open browser.")

    p = sub.add_parser("rename", help="Set or change a program's slug")
    p.add_argument("program", help="Program ID or current slug")
    p.add_argument("new_slug", help="New slug (e.g. 'message-classifier') or empty string to remove")
    p.add_argument("--json", action="store_true", help="JSON output")

    p = sub.add_parser("info", help="Show program info")
    p.add_argument("program", help="Program name or ID")
    p.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0
    if args.command == "run":
        if args.base and args.interpreter is None:
            parser.error("paw run --base requires --interpreter")
        if args.program is not None and args.interpreter is not None:
            parser.error("--interpreter is only valid with --base")

    commands = {
        "compile": cmd_compile,
        "run": cmd_run,
        "login": cmd_login,
        "rename": cmd_rename,
        "info": cmd_info,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
