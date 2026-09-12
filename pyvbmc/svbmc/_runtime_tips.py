"""Process-local scheduling of S-VBMC guidance."""

import random
import threading

from pyvbmc._user_hints import emit_user_hint

from ._tip_catalog import TIPS

_LOCK = threading.Lock()
_RNG = random.Random()
_ORDER = None
_SEEN = set()
_ELIGIBLE_STARTS = 0


def consider_runtime_tip(*, enabled, display, noisy, emitter=emit_user_hint):
    """Emit at most one tip on every third eligible start, beginning at one."""
    global _ORDER, _ELIGIBLE_STARTS
    if not enabled or not display:
        return None
    with _LOCK:
        if _ORDER is None:
            _ORDER = list(TIPS)
            _RNG.shuffle(_ORDER)
        remaining = [
            tip
            for tip in _ORDER
            if tip.id not in _SEEN and (noisy or not tip.noisy_only)
        ]
        if not remaining:
            return None
        opportunity = _ELIGIBLE_STARTS
        _ELIGIBLE_STARTS += 1
        if opportunity % 3:
            return None
        tip = next((tip for tip in remaining if tip.noisy_only), remaining[0])
        if not emitter(f"Tip: {tip.text}", display=display):
            return None
        _SEEN.add(tip.id)
        return tip
