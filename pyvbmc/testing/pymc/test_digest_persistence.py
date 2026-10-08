"""The ordered array fingerprint survives fresh-process recompilation."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

pm = pytest.importorskip("pymc")
pytest.importorskip("arviz_base")
import pytensor
import pytensor.tensor as pt
from pytensor.compile.builders import OpFromGraph

from pyvbmc import VBMC
from pyvbmc.pymc import PyMCTarget


def _target(kind):
    if kind == "garch":
        with pm.Model() as model:
            omega = pm.HalfNormal("omega", 0.5)
            pm.GARCH11(
                "y",
                omega=omega,
                alpha_1=0.2,
                beta_1=0.5,
                initial_vol=1.0,
                observed=np.array([-0.3, 0.1, -0.2, 0.4]),
            )
        return PyMCTarget(
            model,
            start={"omega": 0.2},
            plausible_bounds={"omega": (0.05, 0.4)},
            seed=940,
        )

    with pm.Model() as model:
        x = pm.Normal("x", 0.0, 2.0)
        if kind == "loop":
            weights = np.array([0.43, 0.71, 1.17])
            mean = pytensor.scan(
                lambda i, accumulated, s: accumulated
                + pt.as_tensor_variable(weights)[i] * s,
                sequences=pt.arange(len(weights)),
                outputs_info=pt.zeros(()),
                non_sequences=[x],
                return_updates=False,
            )[-1]
        elif kind == "opfromgraph":
            value = pt.dscalar("value")
            product = OpFromGraph(
                [value],
                [
                    pt.dot(
                        np.array([0.31, 0.67]), pt.stack([value, value**2])
                    )
                ],
            )
            mean = product(x)
        else:
            mean = x
        pm.Normal("y", mean, 1.0, observed=np.array([-0.2, 0.6, 0.8]))
    if kind == "automatic":
        return PyMCTarget(model, seed=940)
    return PyMCTarget(
        model,
        start={"x": 0.1},
        plausible_bounds={"x": (-1.0, 1.0)},
        seed=940,
    )


@pytest.mark.parametrize("kind", ["loop", "garch", "opfromgraph", "automatic"])
def test_digest_survives_fresh_process_load_and_resave(kind, tmp_path):
    target = _target(kind)
    vbmc = VBMC(
        target,
        options={
            "display": "off",
            "performance_calibration": "off",
            "min_iter": 0,
            "max_iter": 1,
            "do_final_boost": False,
        },
        seed=941,
    )
    vbmc.optimize()
    assert vbmc.iteration == 0
    assert vbmc._pymc_target_change(evaluate=True) is None
    source = tmp_path / "run.pkl"
    vbmc.save(source)
    points = np.vstack((target.x0, target.plb, target.pub))
    np.savez(
        tmp_path / "expected.npz",
        points=points,
        values=[target.log_joint(point) for point in points],
        digest=target._constants_digest,
    )

    # Each round starts another interpreter with a different hash seed.
    # Check both the live saved state and restoration from real history.
    code = textwrap.dedent(
        """
        import io
        import logging
        import sys
        import numpy as np
        from pyvbmc import VBMC

        stream = io.StringIO()
        logging.getLogger('VBMC').addHandler(logging.StreamHandler(stream))
        expected = np.load('expected.npz')
        for name in sys.argv[1:]:
            for iteration in (None, 0):
                run = VBMC.load(name, iteration=iteration)
                assert run._pymc_target_changed is None
                assert run._pymc_target_change(evaluate=True) is None
                assert run.target._constants_digest == str(expected['digest'])
                np.testing.assert_allclose(
                    [run.target.log_joint(x) for x in expected['points']],
                    expected['values'], rtol=0, atol=1e-10,
                )
                run.save(f'{name}.{iteration}.pkl')
        assert 'PyMC target of this run no longer matches' not in stream.getvalue()
        """
    )
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = os.pathsep.join(
        [root, env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    paths = [source]
    for hash_seed in (41, 73):
        env["PYTHONHASHSEED"] = str(hash_seed)
        result = subprocess.run(
            [sys.executable, "-c", code, *map(str, paths)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        paths = [
            Path(f"{path}.{iteration}.pkl")
            for path in paths
            for iteration in (None, 0)
        ]
