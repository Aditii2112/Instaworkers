"""Restricted HTTP client tool adapter with domain allowlist.

Only domains explicitly listed in the allowlist can be contacted.
"""

from urllib.parse import urlparse

import httpx

_DEFAULT_TIMEOUT = 30.0


def _check_allowlist(url: str, allowlist: list[str]):
    host = urlparse(url).hostname or ""
    if not any(host == d or host.endswith(f".{d}") for d in allowlist):
        raise PermissionError(
            f"Domain '{host}' not in allowlist. Allowed: {allowlist}"
        )


def http_get(
    url: str,
    headers: dict | None = None,
    allowlist: list[str] | None = None,
) -> dict:
    if allowlist:
        _check_allowlist(url, allowlist)
    resp = httpx.get(url, headers=headers or {}, timeout=_DEFAULT_TIMEOUT)
    return {
        "status_code": resp.status_code,
        "headers": dict(resp.headers),
        "body": resp.text[:10_000],
    }


def http_post(
    url: str,
    body: dict | None = None,
    headers: dict | None = None,
    allowlist: list[str] | None = None,
) -> dict:
    if allowlist:
        _check_allowlist(url, allowlist)
    resp = httpx.post(
        url, json=body, headers=headers or {}, timeout=_DEFAULT_TIMEOUT,
    )
    return {
        "status_code": resp.status_code,
        "headers": dict(resp.headers),
        "body": resp.text[:10_000],
    }
