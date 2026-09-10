"""Small stdout helper for optional user guidance."""

from __future__ import annotations

from collections.abc import Iterable


def emit_user_hint(
    message: str, *, display: bool | str, urls: Iterable[str] = ()
) -> bool:
    """Print a user hint and its URLs when display output is enabled."""
    if not display or display == "off":
        return False

    print(message)
    for url in urls:
        print(url)
    return True
