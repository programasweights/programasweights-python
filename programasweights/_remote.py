from __future__ import annotations

import httpx

from .errors import raise_for_api_status


def _infer(
    client: httpx.Client,
    program_id: str,
    input_text: str,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    if not isinstance(input_text, str):
        raise TypeError("input_text must be a string.")

    payload: dict[str, object] = {"program_id": program_id, "input": input_text}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature

    response = client.post("api/v1/infer", json=payload)
    raise_for_api_status(response)
    data = response.json()
    if not isinstance(data, dict) or not isinstance(data.get("output"), str):
        raise ValueError("PAW inference returned an invalid output.")
    return data["output"]


class RemotePawFunction:
    """A hosted program that owns its HTTP client."""

    def __init__(self, client: httpx.Client, program_id: str):
        self._client = client
        self._program_id = program_id

    def __call__(
        self,
        input_text: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        logits_processor=None,
    ) -> str:
        if self._client.is_closed:
            raise RuntimeError("This RemotePawFunction has been closed.")
        if logits_processor is not None:
            raise ValueError("logits_processor is only supported for local inference.")
        return _infer(
            self._client,
            self._program_id,
            input_text,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> RemotePawFunction:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def load_remote_function(reference: str) -> RemotePawFunction:
    from urllib.parse import quote

    from .cache import is_program_id
    from .config import get_api_key, get_api_url

    headers = {}
    api_key = get_api_key()
    if api_key:
        headers["X-API-Key"] = api_key
    client = httpx.Client(
        base_url=get_api_url().rstrip("/") + "/",
        headers=headers,
        timeout=60.0,
    )
    try:
        program_id = reference
        if not is_program_id(reference):
            response = client.get(
                "api/v1/programs/resolve/" + quote(reference, safe=""),
                timeout=10.0,
            )
            raise_for_api_status(response)
            data = response.json()
            program_id = data.get("program_id") if isinstance(data, dict) else None
            if not isinstance(program_id, str) or not is_program_id(program_id):
                raise ValueError("PAW returned an invalid program ID.")
        return RemotePawFunction(client, program_id)
    except BaseException:
        client.close()
        raise
