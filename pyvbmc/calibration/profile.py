"""Immutable performance-calibration settings.

This module deliberately depends only on the Python standard library.  Cache
and campaign code may import it, while numerical objects can store profiles
without creating import cycles or performing machine inspection.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, ClassVar, Optional

DEFAULT_CHUNK_ELEMENTS = 2**16
PROFILE_SCHEMA_VERSION = 1


class _FrozenMapping(Mapping):
    """A small, pickle-friendly immutable mapping of JSON scalar values."""

    __slots__ = ("_items",)

    def __init__(self, values: Optional[Mapping[str, Any]] = None):
        if values is None:
            values = {}
        if not isinstance(values, Mapping):
            raise TypeError("provenance must be a mapping.")

        items = []
        for key, value in values.items():
            if not isinstance(key, str) or not key:
                raise ValueError("provenance keys must be nonempty strings.")
            if not isinstance(value, (str, bool, int, float, type(None))):
                raise ValueError(
                    "provenance values must be JSON scalar values."
                )
            if isinstance(value, Real) and not isinstance(value, Integral):
                if not math.isfinite(value):
                    raise ValueError("provenance numbers must be finite.")
            items.append((key, value))
        object.__setattr__(self, "_items", tuple(sorted(items)))

    def __setattr__(self, name, value):
        raise TypeError("Calibration profile provenance is immutable.")

    def __reduce__(self):
        return self.__class__, (dict(self._items),)

    def __getitem__(self, key):
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(self._items)

    def __repr__(self) -> str:
        return repr(dict(self._items))


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer.")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _optional_string(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string or None.")
    return value


@dataclass(frozen=True)
class CalibrationProfile:
    """Fixed chunk settings selected for one PyVBMC run.

    The profile stores only compact provenance. Detailed measurements belong
    to the machine-local calibration report rather than posterior histories.

    Parameters
    ----------
    pdf_chunk_elements : int, optional
        Element budget for posterior density evaluation.
    entropy_grad_chunk_elements : int, optional
        Element budget for Monte Carlo entropy calls that request gradients.
    entropy_value_chunk_elements : int, optional
        Element budget for value-only Monte Carlo entropy calls.
    source : str, optional
        How the settings were obtained, such as ``"default"``, ``"cache"``
        or ``"explicit"`` (the default for a directly constructed profile).
    status : str, optional
        Completion or fallback status associated with the settings.
    fingerprint : str or None, optional
        Compatible machine/configuration fingerprint, when available.
    provenance : mapping, optional
        Compact string-keyed metadata with JSON scalar values.
    cache_path : str or None, optional
        Calibration report path, when the profile was persisted.
    """

    schema_version: ClassVar[int] = PROFILE_SCHEMA_VERSION

    pdf_chunk_elements: int = DEFAULT_CHUNK_ELEMENTS
    entropy_grad_chunk_elements: int = DEFAULT_CHUNK_ELEMENTS
    entropy_value_chunk_elements: int = DEFAULT_CHUNK_ELEMENTS
    source: str = "explicit"
    status: str = "complete"
    fingerprint: Optional[str] = None
    provenance: Mapping[str, Any] = _FrozenMapping()
    cache_path: Optional[str] = None

    def __post_init__(self):
        for name in (
            "pdf_chunk_elements",
            "entropy_grad_chunk_elements",
            "entropy_value_chunk_elements",
        ):
            object.__setattr__(
                self, name, _positive_integer(getattr(self, name), name)
            )
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a nonempty string.")
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("status must be a nonempty string.")
        object.__setattr__(
            self,
            "fingerprint",
            _optional_string(self.fingerprint, "fingerprint"),
        )
        object.__setattr__(
            self, "cache_path", _optional_string(self.cache_path, "cache_path")
        )
        if not isinstance(self.provenance, _FrozenMapping):
            object.__setattr__(
                self, "provenance", _FrozenMapping(self.provenance)
            )

    @property
    def settings(self) -> dict[str, int]:
        """Return the three numerical settings as a new dictionary."""
        return {
            "pdf_chunk_elements": self.pdf_chunk_elements,
            "entropy_grad_chunk_elements": self.entropy_grad_chunk_elements,
            "entropy_value_chunk_elements": self.entropy_value_chunk_elements,
        }

    @property
    def uses_historical_defaults(self) -> bool:
        """Whether all numerical settings equal the historical constants."""
        return all(
            value == DEFAULT_CHUNK_ELEMENTS for value in self.settings.values()
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this profile to its versioned, JSON-compatible schema."""
        return {
            "schema_version": self.schema_version,
            "settings": self.settings,
            "source": self.source,
            "status": self.status,
            "fingerprint": self.fingerprint,
            "provenance": dict(self.provenance),
            "cache_path": self.cache_path,
        }

    def __deepcopy__(self, memo):
        """Immutable profiles are safely shared by posterior copies."""
        return self

    def __reduce__(self):
        # Include the versioned representation in every enclosing VP/VBMC
        # pickle. A later schema cannot silently reinterpret older fields.
        return _profile_from_serialized_dict, (self.to_dict(),)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CalibrationProfile":
        """Validate and deserialize a profile from its versioned schema."""
        if not isinstance(data, Mapping):
            raise TypeError("Calibration profile data must be a mapping.")
        expected = {
            "schema_version",
            "settings",
            "source",
            "status",
            "fingerprint",
            "provenance",
            "cache_path",
        }
        if set(data) != expected:
            raise ValueError(
                "Calibration profile fields do not match schema v1."
            )
        schema_version = data["schema_version"]
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, Integral)
            or int(schema_version) != PROFILE_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported calibration profile schema version.")
        settings = data["settings"]
        if not isinstance(settings, Mapping):
            raise ValueError("Calibration profile settings must be a mapping.")
        setting_names = {
            "pdf_chunk_elements",
            "entropy_grad_chunk_elements",
            "entropy_value_chunk_elements",
        }
        if set(settings) != setting_names:
            raise ValueError(
                "Calibration profile settings do not match schema v1."
            )
        return cls(
            **{name: settings[name] for name in setting_names},
            source=data["source"],
            status=data["status"],
            fingerprint=data["fingerprint"],
            provenance=data["provenance"],
            cache_path=data["cache_path"],
        )


def default_profile(
    *,
    source: str = "default",
    status: str = "complete",
    fingerprint: Optional[str] = None,
    cache_path: Optional[str] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> CalibrationProfile:
    """Create a historical-default profile with compact metadata."""
    return CalibrationProfile(
        source=source,
        status=status,
        fingerprint=fingerprint,
        cache_path=cache_path,
        provenance={} if provenance is None else provenance,
    )


def _profile_from_serialized_dict(data):
    return CalibrationProfile.from_dict(data)
