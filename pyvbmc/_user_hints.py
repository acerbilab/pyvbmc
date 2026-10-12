"""Small stdout helper for optional user guidance."""

from __future__ import annotations

from collections.abc import Iterable


def emit_user_hint(
    message: str, *, display: bool | str, urls: Iterable[str] = ()
) -> bool:
    """Print a user hint and its URLs when display output is enabled.

    Returns whether the hint's message printed. An output stream that fails
    (a closed pipe, say) raises nothing, so that a hint cannot stop a run:
    when the message cannot be printed the call returns ``False``, and a URL
    that cannot be printed after it is left out.
    """
    if not display or display == "off":
        return False

    try:
        print(message)
    except Exception:
        return False
    for url in urls:
        try:
            print(url)
        except Exception:
            break
    return True
