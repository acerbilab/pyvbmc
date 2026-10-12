"""Public imports and missing-extra errors, also run without extras."""

import subprocess
import sys
import textwrap

import pytest

from pyvbmc import VariationalPosterior


def test_imports_are_lazy_in_fresh_process(tmp_path):
    code = """
        import sys
        import pyvbmc
        vp = pyvbmc.VariationalPosterior(2, rng=9)
        vp.sample(2)
        assert not any(name.split('.')[0] in {'torch', 'arviz', 'arviz_base'}
                       for name in sys.modules)
    """
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "module,method", [("torch", "to_torch"), ("arviz_base", "to_arviz")]
)
def test_missing_extra_has_actionable_error(tmp_path, module, method):
    code = """
        import importlib.abc
        import sys
        from pyvbmc import VariationalPosterior
        class Missing(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] == MODULE:
                    raise ModuleNotFoundError('blocked for test', name=MODULE)
        sys.meta_path.insert(0, Missing())
        vp = VariationalPosterior(1, rng=9)
        state = vp.rng.bit_generator.state
        try:
            getattr(vp, METHOD)()
        except ImportError as exc:
            assert 'pyvbmc[' in str(exc), str(exc)
        else:
            raise AssertionError('expected missing-extra error')
        assert vp.rng.bit_generator.state == state
    """
    code = f"MODULE={module!r}; METHOD={method!r}\n" + textwrap.dedent(code)
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_arviz_on_older_python_is_explicit(monkeypatch):
    vp = VariationalPosterior(1, rng=4)
    state = vp.rng.bit_generator.state
    with monkeypatch.context() as patch:
        patch.setattr(sys, "version_info", (3, 10, 0))
        with pytest.raises(ImportError, match="Python 3.12"):
            vp.to_arviz(2)
    assert vp.rng.bit_generator.state == state
