"""Exercise native campaign locks across actual process boundaries."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pyvbmc.calibration import _cache
from pyvbmc.testing.calibration.test_cache import make_test_record


@pytest.fixture(autouse=True)
def _shared_identity(monkeypatch):
    # These tests exercise OS locks, not backend discovery. In particular,
    # pytest may have loaded optional native libraries absent from the child.
    identity = _cache.machine_identity()
    identity["cpu_model"] = "Process lock test CPU"
    identity["architecture"] = "test64"
    backend = {name: None for name in _cache._BACKEND_FIELDS}
    backend.update(internal_api="test-blas", version="1", num_threads=1)
    identity["backends"] = [backend]
    monkeypatch.setattr(_cache, "machine_identity", lambda: identity)
    monkeypatch.setenv(
        "PYVBMC_TEST_CALIBRATION_IDENTITY", json.dumps(identity)
    )


def _child(script):
    environment = os.environ.copy()
    root = Path(__file__).resolve().parents[3]
    environment["PYTHONPATH"] = str(root)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, os\n"
            "from pyvbmc.calibration import _cache\n"
            "_cache.machine_identity = lambda: json.loads("
            "os.environ['PYVBMC_TEST_CALIBRATION_IDENTITY'])\n" + script,
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )


def test_other_process_reads_cache_without_waiting_for_campaign(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("PYVBMC_CACHE_DIR", str(tmp_path))
    identity = _cache.machine_identity()
    record = make_test_record(
        identity, **{name: 2**14 for name in _cache._SETTING_NAMES}
    )
    _cache.write_record(record)
    with _cache.campaign_guard() as guard:
        assert guard.acquired and guard.persistent
        completed = _child(
            "import json\n"
            "from pyvbmc.calibration import _cache\n"
            "profile = _cache.resolve_cached_profile()\n"
            "with _cache.campaign_guard() as guard:\n"
            "    print(json.dumps({'acquired': guard.acquired, "
            "'source': profile.source, 'budget': "
            "profile.pdf_chunk_elements}))\n"
        )
    assert json.loads(completed.stdout) == {
        "acquired": False,
        "source": "cache",
        "budget": 2**14,
    }


def test_native_lock_releases_when_owner_exits_abruptly(monkeypatch, tmp_path):
    monkeypatch.setenv("PYVBMC_CACHE_DIR", str(tmp_path))
    completed = _child(
        "import os\n"
        "from pyvbmc.calibration import _cache\n"
        "with _cache.campaign_guard() as guard:\n"
        "    assert guard.acquired and guard.persistent\n"
        "    print('acquired', flush=True)\n"
        "    os._exit(0)\n"
    )
    assert completed.stdout.strip() == "acquired"
    with _cache.campaign_guard() as guard:
        assert guard.acquired and guard.persistent
