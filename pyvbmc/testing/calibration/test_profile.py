import copy
from dataclasses import FrozenInstanceError

import dill
import pytest

from pyvbmc.calibration.profile import (
    DEFAULT_CHUNK_ELEMENTS,
    CalibrationProfile,
    default_profile,
)


def test_default_profile_and_settings_are_immutable():
    profile = default_profile(provenance={"reason": "historical"})

    assert profile.settings == {
        "pdf_chunk_elements": DEFAULT_CHUNK_ELEMENTS,
        "entropy_grad_chunk_elements": DEFAULT_CHUNK_ELEMENTS,
        "entropy_value_chunk_elements": DEFAULT_CHUNK_ELEMENTS,
    }
    assert profile.uses_historical_defaults
    with pytest.raises(FrozenInstanceError):
        profile.pdf_chunk_elements = 1
    with pytest.raises(TypeError):
        profile.provenance["reason"] = "changed"
    with pytest.raises(TypeError):
        profile.provenance._items = ()
    settings = profile.settings
    settings["pdf_chunk_elements"] = 1
    assert profile.pdf_chunk_elements == DEFAULT_CHUNK_ELEMENTS


@pytest.mark.parametrize(
    "value",
    [True, False, 0, -1, 1.5, "65536"],
)
def test_profile_rejects_invalid_element_budgets(value):
    with pytest.raises(ValueError, match="positive integer"):
        CalibrationProfile(pdf_chunk_elements=value)


def test_profile_roundtrip_copy_and_pickle():
    profile = CalibrationProfile(
        pdf_chunk_elements=7,
        entropy_grad_chunk_elements=11,
        entropy_value_chunk_elements=13,
        source="explicit",
        status="complete",
        fingerprint="abc",
        provenance={"version": "1.5", "elapsed": 1.25, "saved": True},
        cache_path="/cache/profile.json",
    )

    assert CalibrationProfile.from_dict(profile.to_dict()) == profile
    assert copy.deepcopy(profile) is profile
    assert dill.loads(dill.dumps(profile)) == profile
    factory, args = profile.__reduce__()
    assert args[0]["schema_version"] == 1
    assert factory(*args) == profile
    assert not profile.uses_historical_defaults


def test_manual_profile_has_explicit_provenance():
    assert CalibrationProfile(pdf_chunk_elements=16384).source == "explicit"
    assert default_profile().source == "default"


def test_profile_schema_is_strict():
    data = default_profile().to_dict()
    data["schema_version"] = True
    with pytest.raises(ValueError, match="schema version"):
        CalibrationProfile.from_dict(data)

    data = default_profile().to_dict()
    data["extra"] = "field"
    with pytest.raises(ValueError, match="fields"):
        CalibrationProfile.from_dict(data)

    data = default_profile().to_dict()
    data["settings"]["pdf_chunk_elements"] = 1.0
    with pytest.raises(ValueError, match="positive integer"):
        CalibrationProfile.from_dict(data)


def test_profile_rejects_noncompact_provenance():
    with pytest.raises(ValueError, match="JSON scalar"):
        CalibrationProfile(provenance={"timings": [1.0, 2.0]})
