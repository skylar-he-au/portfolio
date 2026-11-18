"""Stateless sessions and trusted Worker headers; no database access."""

import base64
import hashlib
import hmac
import ipaddress
import re
import secrets
import time
from dataclasses import dataclass

from fastapi import Request

from app.config import Settings

REJECTIONS = {
    "forbidden": (403, "Something went wrong — please try again."),
    "auth": (401, "Please try again."),
    "slow_down": (429, "Too many requests — please wait a minute and try again."),
    "limit_day": (429, "Today's AI limit is reached — please email me"),
    "paused_month": (503, "The AI is paused until next month — please email me"),
    "unavailable": (
        503,
        "The assistant is unavailable right now — please try again later or email me.",
    ),
}


class Rejected(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        self.status_code, self.message = REJECTIONS[code]
        super().__init__(code)


@dataclass(frozen=True)
class Session:
    token_key: bytes
    ip: str


def _signature(secret: str, payload: str) -> str:
    digest = hmac.digest(secret.encode(), payload.encode("ascii"), "sha256")
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def sign_token(secret: str, now: int | float) -> tuple[str, int]:
    exp = int(now) + 24 * 60 * 60
    payload = f"v1.{secrets.token_hex(16)}.{exp}"
    return f"{payload}.{_signature(secret, payload)}", exp


def verify_token(secret: str, token: str, now: int | float) -> bytes | None:
    parts = token.split(".")
    if len(parts) != 4:
        return None
    version, token_id, exp, signature = parts
    if (
        version != "v1"
        or re.fullmatch(r"[0-9a-f]{32}", token_id) is None
        or re.fullmatch(r"[0-9]+", exp) is None
        or len(exp) > 20
        or re.fullmatch(r"[A-Za-z0-9_-]{43}", signature) is None
    ):
        return None
    if int(exp) <= now or not hmac.compare_digest(
        signature, _signature(secret, ".".join(parts[:3]))
    ):
        return None
    return hashlib.sha256(token_id.encode("ascii")).digest()


def require_origin(request: Request) -> str:
    """Trust only headers supplied by the authenticated Worker."""
    settings: Settings = request.app.state.settings
    expected = settings.origin_secret
    supplied = request.headers.get("X-Origin-Secret", "")
    if expected is None or not hmac.compare_digest(
        supplied.encode(), expected.get_secret_value().encode()
    ):
        raise Rejected("forbidden")
    try:
        raw = request.headers.get("X-Client-IP", "")
        # Zone IDs are interface names, not public visitor addresses.
        if "%" in raw:
            raise ValueError("scoped address")
        address = ipaddress.ip_address(raw)
    except ValueError:
        raise Rejected("forbidden") from None
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return str(address.ipv4_mapped)
        return str(ipaddress.ip_network(f"{address}/64", strict=False))
    return str(address)


def require_session(request: Request) -> Session:
    ip = require_origin(request)
    settings: Settings = request.app.state.settings
    authorization = request.headers.get("Authorization", "").split()
    if (
        settings.session_secret is None
        or len(authorization) != 2
        or authorization[0].lower() != "bearer"
    ):
        raise Rejected("auth")
    key = verify_token(settings.session_secret.get_secret_value(), authorization[1], time.time())
    if key is None:
        raise Rejected("auth")
    return Session(key, ip)
