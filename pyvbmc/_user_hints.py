"""Small stdout helper for optional user guidance."""

from __future__ import annotations

from collections.abc import Iterable


def emit_user_hint(
    message: str, *, display: bool | str, urls: Iterable[str] = ()
) -> bool:
    """Print a user hint and its URLs when display output is enabled.

    Returns whether the hint printed. A failing output stream, such as a
    closed pipe, prints nothing and returns ``False``, so that a hint cannot
    stop a run.
    """
    if not display or display == "off":
        return False

    try:
        print(message)
        for url in urls:
            print(url)
    except Exception:
        return False
    return True
