"""Import hygiene for the optional PyMC integration."""

import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest


def _run(code, cwd, *, env=None):
    process = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )
    assert process.returncode == 0, process.stderr


def test_pymc_namespaces_do_not_import_optional_dependencies(tmp_path):
    _run(
        """
        import sys
        import pyvbmc
        import pyvbmc.pymc

        loaded = {name.split('.')[0] for name in sys.modules}
        assert 'pymc' not in loaded
        assert 'pytensor' not in loaded
        assert 'PyMCTarget' in dir(pyvbmc)
        assert {'PyMCTarget', 'UnsupportedModel'} <= set(dir(pyvbmc.pymc))
        """,
        tmp_path,
    )


def test_root_pymc_target_attribute_is_lazy(tmp_path):
    _run(
        """
        import sys
        import pyvbmc

        assert 'pyvbmc.pymc' not in sys.modules
        target_type = pyvbmc.PyMCTarget
        import pyvbmc.pymc
        assert target_type is pyvbmc.pymc.PyMCTarget
        loaded = {name.split('.')[0] for name in sys.modules}
        assert 'pymc' not in loaded
        assert 'pytensor' not in loaded

        try:
            pyvbmc.pymc.no_such_attribute
        except AttributeError as exc:
            assert 'no_such_attribute' in str(exc), str(exc)
        else:
            raise AssertionError('expected AttributeError')
        """,
        tmp_path,
    )


def test_ordinary_vbmc_construction_does_not_import_pymc(tmp_path):
    _run(
        """
        import sys
        import numpy as np
        from pyvbmc import VBMC

        vbmc = VBMC(
            np.sum,
            np.array([[0.0]]),
            np.array([[-2.0]]),
            np.array([[2.0]]),
            np.array([[-1.0]]),
            np.array([[1.0]]),
            options={'display': 'off'},
            seed=1,
        )
        assert vbmc.target is None
        loaded = {name.split('.')[0] for name in sys.modules}
        assert 'pymc' not in loaded
        assert 'pytensor' not in loaded
        """,
        tmp_path,
    )


def test_missing_pymc_extra_has_actionable_error(tmp_path):
    _run(
        r"""
        import importlib.abc
        import sys

        import pyvbmc

        class Missing(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'pymc', 'pytensor'}:
                    raise ModuleNotFoundError(
                        'blocked for test', name=fullname.split('.')[0]
                    )

        sys.meta_path.insert(0, Missing())
        try:
            pyvbmc.PyMCTarget(object())
        except ImportError as exc:
            assert 'pyvbmc[pymc]' in str(exc), str(exc)
        else:
            raise AssertionError('expected a missing-extra ImportError')
        """,
        tmp_path,
    )


def test_float32_pytensor_configuration_is_rejected(tmp_path):
    if importlib.util.find_spec("pymc") is None:
        pytest.skip("PyMC is not installed")
    env = os.environ.copy()
    env["PYTENSOR_FLAGS"] = "floatX=float32"
    _run(
        """
        import pymc as pm
        from pyvbmc import PyMCTarget

        with pm.Model() as model:
            pm.Normal('x')
        try:
            PyMCTarget(model, seed=1)
        except ValueError as exc:
            message = str(exc)
            assert 'floatX is float32' in message, message
            assert 'float64' in message, message
        else:
            raise AssertionError('expected a float64-requirement error')
        """,
        tmp_path,
        env=env,
    )
