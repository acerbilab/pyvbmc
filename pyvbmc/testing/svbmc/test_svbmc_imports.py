"""Import hygiene of ``pyvbmc.svbmc``; runs without the torch extra.

Importing PyVBMC, the S-VBMC subpackage or its helpers must not import
torch: the dependency is optional and only :class:`~pyvbmc.svbmc.SVBMC`
needs it, at construction. Each check runs in a fresh interpreter, so an
import made by another test cannot hide a regression.
"""

import subprocess
import sys
import textwrap


def _run(code, cwd):
    """Run ``code`` in a fresh interpreter; fail with its stderr."""
    process = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    return process


def test_importing_svbmc_does_not_import_torch(tmp_path):
    code = """
        import sys
        import pyvbmc
        import pyvbmc.svbmc
        from pyvbmc.svbmc import utils

        assert callable(utils.find_init_bounds)
        assert callable(utils.overlay_corner_plot)
        loaded = [n for n in sys.modules if n.split('.')[0] == 'torch']
        assert not loaded, loaded
    """
    _run(code, tmp_path)


def test_root_attribute_is_lazy(tmp_path):
    code = """
        import sys
        import pyvbmc

        assert 'pyvbmc.svbmc' not in sys.modules
        assert pyvbmc.SVBMC is not None
        import pyvbmc.svbmc
        assert pyvbmc.SVBMC is pyvbmc.svbmc.SVBMC
        loaded = [n for n in sys.modules if n.split('.')[0] == 'torch']
        assert not loaded, loaded

        try:
            pyvbmc.no_such_attribute
        except AttributeError as exc:
            assert 'no_such_attribute' in str(exc), str(exc)
        else:
            raise AssertionError('expected AttributeError')
    """
    _run(code, tmp_path)


def test_missing_torch_extra_has_actionable_error(tmp_path):
    code = """
        import importlib.abc
        import sys

        import numpy as np
        import pyvbmc

        class Missing(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] == 'torch':
                    raise ModuleNotFoundError('blocked for test', name='torch')

        sys.meta_path.insert(0, Missing())

        vp = pyvbmc.VariationalPosterior(2, rng=1)
        vp.stats = {
            'stable': True,
            'elbo': 0.0,
            'I_sk': np.zeros((3, 2)),
            'J_sjk': np.zeros((3, 2, 2)),
        }
        try:
            pyvbmc.SVBMC([vp])
        except ImportError as exc:
            assert 'pyvbmc[torch]' in str(exc), str(exc)
        else:
            raise AssertionError('expected a missing-extra ImportError')

        loaded = [n for n in sys.modules if n.split('.')[0] == 'torch']
        assert not loaded, loaded
    """
    _run(code, tmp_path)
