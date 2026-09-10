"""Process-local scheduling for optional runtime tips."""

from __future__ import annotations

import random
import threading
from collections.abc import Callable, Sequence

from pyvbmc._user_hints import emit_user_hint

from ._tip_catalog import TIPS, Tip

_TIP_FREQUENCIES = frozenset({"normal", "low_frequency"})
_STATE_LOCK = threading.Lock()
_RNG = random.Random()
_ORDER: list[Tip] | None = None
_SEEN_IDS: set[str] = set()
_ELIGIBLE_STARTS = 0
_LAST_FREQUENCY: str | None = None


def _validate_catalog(catalog: Sequence[Tip]) -> None:
    """Validate the fields the scheduler depends on."""
    seen_ids: set[str] = set()
    for tip in catalog:
        if not isinstance(tip, Tip):
            raise TypeError("Runtime tip catalog entries must be Tip objects.")
        if not tip.id or tip.id in seen_ids:
            raise ValueError("Runtime tip IDs must be nonempty and unique.")
        if not tip.text:
            raise ValueError("Runtime tip text must be nonempty.")
        if tip.frequency not in _TIP_FREQUENCIES:
            raise ValueError(
                "Runtime tip frequency must be 'normal' or 'low_frequency'."
            )
        if not isinstance(tip.urls, tuple) or not all(
            isinstance(url, str) and url for url in tip.urls
        ):
            raise ValueError(
                "Runtime tip URLs must be nonempty strings in a tuple."
            )
        seen_ids.add(tip.id)


def _remaining_order(catalog: Sequence[Tip], rng: random.Random) -> list[Tip]:
    """Return the process order, constructing it at the first opportunity."""
    global _ORDER
    if _ORDER is None:
        _validate_catalog(catalog)
        _ORDER = list(catalog)
        rng.shuffle(_ORDER)
    return [tip for tip in _ORDER if tip.id not in _SEEN_IDS]


def _next_tip(remaining: Sequence[Tip]) -> Tip:
    """Choose the next tip, spacing low-frequency entries when possible."""
    first = remaining[0]
    if (
        _LAST_FREQUENCY == "low_frequency"
        and first.frequency == "low_frequency"
    ):
        for tip in remaining[1:]:
            if tip.frequency == "normal":
                return tip
    return first


def consider_runtime_tip(
    *,
    display: bool | str,
    enabled: bool,
    calibration_reminder_emitted: bool,
    catalog: Sequence[Tip] = TIPS,
    rng: random.Random | None = None,
    emitter: Callable[..., bool] = emit_user_hint,
) -> Tip | None:
    """Consider and possibly emit one tip for an eligible new-run start."""
    global _ELIGIBLE_STARTS, _LAST_FREQUENCY

    if not display or display == "off" or not enabled:
        return None
    if calibration_reminder_emitted:
        return None

    with _STATE_LOCK:
        private_rng = _RNG if rng is None else rng
        remaining = _remaining_order(catalog, private_rng)
        if not remaining:
            return None

        opportunity = _ELIGIBLE_STARTS
        _ELIGIBLE_STARTS += 1
        if opportunity % 3 != 0:
            return None

        tip = _next_tip(remaining)
        if not emitter(f"Tip: {tip.text}", display=display, urls=tip.urls):
            return None
        _SEEN_IDS.add(tip.id)
        _LAST_FREQUENCY = tip.frequency
        return tip


def _reset_runtime_tip_state(*, rng: random.Random | None = None) -> None:
    """Reset the process-local scheduler for isolated tests."""
    global _RNG, _ORDER, _ELIGIBLE_STARTS, _LAST_FREQUENCY
    with _STATE_LOCK:
        _RNG = random.Random() if rng is None else rng
        _ORDER = None
        _SEEN_IDS.clear()
        _ELIGIBLE_STARTS = 0
        _LAST_FREQUENCY = None
