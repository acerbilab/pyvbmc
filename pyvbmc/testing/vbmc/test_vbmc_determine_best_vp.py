import numpy as np

from pyvbmc import VBMC


def create_vbmc(
    D: int,
    x0: float,
    lower_bounds: float,
    upper_bounds: float,
    plausible_lower_bounds: float,
    plausible_upper_bounds: float,
    options: dict = None,
):
    fun = lambda x: np.sum(x + 2)
    lb = np.ones((1, D)) * lower_bounds
    ub = np.ones((1, D)) * upper_bounds
    x0_array = np.ones((2, D)) * x0
    plb = np.ones((1, D)) * plausible_lower_bounds
    pub = np.ones((1, D)) * plausible_upper_bounds
    return VBMC(fun, x0_array, lb, ub, plb, pub, options)


def recorded_history(vbmc, elbo, elbo_sd, r_index, stable):
    """Install an iteration history shaped as a run records one.

    ``IterationHistory`` stores object-dtype arrays, so the flags and the
    scores come back as Python objects rather than as native dtypes.
    """
    n = len(elbo)
    vbmc.vp.stats = dict()
    vbmc.iteration_history["iter"] = np.array(range(n), dtype=object)
    vbmc.iteration_history["vp"] = np.array([vbmc.vp] * n, dtype=object)
    vbmc.iteration_history["elbo"] = np.array(elbo, dtype=object)
    vbmc.iteration_history["elbo_sd"] = np.array(elbo_sd, dtype=object)
    vbmc.iteration_history["r_index"] = np.array(r_index, dtype=object)
    vbmc.iteration_history["stable"] = np.array(stable, dtype=object)


def assert_copy_of_recorded(vp, recorded):
    """The returned posterior is a copy of the recorded one, on the same
    random stream and with the same variational parameters."""
    assert vp is not recorded
    assert vp.rng is recorded.rng
    assert vp.K == recorded.K
    for parameter in ("mu", "sigma", "lambd", "w"):
        assert np.array_equal(
            getattr(vp, parameter), getattr(recorded, parameter)
        )


def test_determine_best_vp_returns_a_copy_and_leaves_the_history_alone():
    """The selection returns a copy of the recorded posterior of the
    selected iteration, with that iteration's stability flag written into
    the copy. The recorded posterior keeps the statistics it was recorded
    with, so the history still describes the run as it happened."""
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    recorded_history(
        vbmc,
        elbo=[0.0, 1.0, 2.0],
        elbo_sd=[0.0] * 3,
        r_index=[1.0] * 3,
        stable=[True] * 3,
    )
    recorded = vbmc.iteration_history["vp"][2]
    assert "stable" not in recorded.stats

    vp, __, __, idx_best = vbmc.determine_best_vp()

    assert idx_best == 2
    assert_copy_of_recorded(vp, recorded)
    assert vp.stats["stable"]
    assert "stable" not in recorded.stats


def test_determine_best_vp_receives_the_option_values():
    """The options that govern the selection reach ``determine_best_vp``,
    so setting one of them changes which posterior a run returns."""
    D = 2
    options = {
        "max_iter": 2,
        "max_fun_evals": 40,
        "display": "off",
        "plot": False,
        "print_iteration_header": False,
        # All three away from their defaults (True, 5, 0.25).
        "rank_criterion": False,
        "best_safe_sd": 3,
        "best_frac_back": 0.5,
    }
    vbmc = VBMC(
        lambda x: -0.5 * np.sum(x**2),
        np.zeros((1, D)),
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.full((1, D), -1.0),
        np.full((1, D), 1.0),
        options=options,
        seed=1234,
    )
    calls = []
    unwired = VBMC.determine_best_vp

    def record_call(self, *args, **kwargs):
        calls.append(kwargs)
        return unwired(self, *args, **kwargs)

    vbmc.determine_best_vp = record_call.__get__(vbmc, VBMC)
    vbmc.optimize()

    assert len(calls) > 0
    for kwargs in calls:
        assert kwargs["rank_criterion_flag"] is False
        assert kwargs["safe_sd"] == 3
        assert kwargs["frac_back"] == 0.5


def test_determine_best_vp_ranks_a_recorded_history():
    """The ranking criterion runs on the object-dtype arrays a run
    records, and prefers the recent, high-ELCBO, reliable iteration."""
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    recorded_history(
        vbmc,
        elbo=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        elbo_sd=[0.0] * 6,
        r_index=[5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        stable=[False] * 6,
    )
    __, __, __, idx_best = vbmc.determine_best_vp(rank_criterion_flag=True)
    assert idx_best == 5


def test_best_safe_sd_changes_the_selection():
    """The ELCBO penalty picks a different iteration when the best ELBO
    is also the most uncertain one."""
    scores = dict(
        elbo=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        elbo_sd=[0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        r_index=[0.0] * 6,
        stable=[False] * 6,
    )
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    recorded_history(vbmc, **scores)
    __, __, __, idx_unpenalized = vbmc.determine_best_vp(safe_sd=0)

    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    recorded_history(vbmc, **scores)
    __, __, __, idx_penalized = vbmc.determine_best_vp(safe_sd=5)

    assert idx_unpenalized == 5
    assert idx_penalized == 4


def test_determine_best_vp_last_stable():
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    vbmc.iteration_history["iter"] = np.arange(0, 3)
    vbmc.iteration_history["stable"] = np.full((3), True)
    vbmc.vp.stats = dict()
    vbmc.iteration_history["vp"] = np.array([vbmc.vp, vbmc.vp, vbmc.vp])
    vbmc.iteration_history["elbo"] = np.arange(0, 3)
    vbmc.iteration_history["elbo_sd"] = np.arange(0, 3)
    vp, elbo, elbo_sd, idx_best = vbmc.determine_best_vp()
    assert idx_best == 2
    assert_copy_of_recorded(vp, vbmc.iteration_history["vp"][idx_best])
    assert elbo == 2
    assert elbo_sd == 2


def test_determine_best_vp_rank_criterion_elbo():
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    n_iterations = 3000
    vbmc.iteration_history["iter"] = np.arange(0, n_iterations)
    vbmc.iteration_history["stable"] = np.full((n_iterations), False)
    vbmc.iteration_history["vp"] = np.arange(0, n_iterations)
    vbmc.vp.stats = dict()
    vbmc.iteration_history["vp"] = np.full((n_iterations), vbmc.vp)
    vbmc.iteration_history["elbo"] = np.arange(0, n_iterations)
    vbmc.iteration_history["elbo_sd"] = np.zeros(n_iterations)
    vbmc.iteration_history["r_index"] = np.arange(0, n_iterations)
    vp, elbo, elbo_sd, idx_best = vbmc.determine_best_vp(
        rank_criterion_flag=True
    )
    assert idx_best == n_iterations - 1
    assert_copy_of_recorded(vp, vbmc.iteration_history["vp"][idx_best])
    assert elbo == n_iterations - 1
    assert elbo_sd == 0


def test_determine_best_vp_rank_criterion_max_idx():
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    n_iterations = 3000
    vbmc.iteration_history["iter"] = np.arange(0, n_iterations)
    vbmc.iteration_history["stable"] = np.full((n_iterations), False)
    vbmc.vp.stats = dict()
    vbmc.iteration_history["vp"] = np.full((n_iterations), vbmc.vp)
    vbmc.iteration_history["elbo"] = np.arange(0, n_iterations)
    vbmc.iteration_history["elbo_sd"] = np.zeros(n_iterations)
    vbmc.iteration_history["r_index"] = np.arange(0, n_iterations)
    vp, elbo, elbo_sd, idx_best = vbmc.determine_best_vp(
        rank_criterion_flag=True, max_idx=1000
    )
    assert idx_best == 1000
    assert_copy_of_recorded(vp, vbmc.iteration_history["vp"][idx_best])
    assert elbo == 1000
    assert elbo_sd == 0


def test_determine_best_vp_no_rank_criterion_second_last_stable():
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    n_iterations = 3000
    vbmc.iteration_history["iter"] = np.arange(0, n_iterations)
    vbmc.iteration_history["stable"] = np.full((n_iterations), True)
    vbmc.iteration_history["stable"][1000] = False
    vbmc.vp.stats = dict()
    vbmc.iteration_history["vp"] = np.full((n_iterations), vbmc.vp)
    vbmc.iteration_history["elbo"] = np.arange(0, n_iterations)
    vbmc.iteration_history["elbo_sd"] = np.zeros(n_iterations)
    vbmc.iteration_history["r_index"] = np.arange(0, n_iterations)
    vp, elbo, elbo_sd, idx_best = vbmc.determine_best_vp(
        rank_criterion_flag=False, max_idx=1000
    )
    assert idx_best == 1000
    assert_copy_of_recorded(vp, vbmc.iteration_history["vp"][idx_best])
    assert elbo == 1000
    assert elbo_sd == 0


def test_determine_best_vp_no_rank_criterion_no_stable():
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    n_iterations = 3000
    vbmc.iteration_history["iter"] = np.arange(0, n_iterations)
    vbmc.iteration_history["stable"] = np.full((n_iterations), False)
    vbmc.vp.stats = dict()
    vbmc.iteration_history["vp"] = np.full((n_iterations), vbmc.vp)
    vbmc.iteration_history["elbo"] = np.arange(0, n_iterations)
    vbmc.iteration_history["elbo_sd"] = np.zeros(n_iterations)
    vp, elbo, elbo_sd, idx_best = vbmc.determine_best_vp(
        rank_criterion_flag=False, max_idx=1000
    )
    assert idx_best == 1000
    assert_copy_of_recorded(vp, vbmc.iteration_history["vp"][idx_best])
    assert elbo == 1000
    assert elbo_sd == 0


def test_look_back_window_counts_iterations():
    """Without a stable iteration, the search for the best ELCBO goes back
    over ``ceil(n * frac_back)`` iterations before the last one, ``n``
    being the number of iterations ranked (MATLAB VBMC's ``best_vbmc``:
    ``idx_start = max(1, n - ceil(n * FracBack))`` in 1-based indices)."""
    for n in range(1, 41):
        for frac_back in (0.1, 0.25, 0.5):
            vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
            # A decreasing ELBO makes the earliest iteration of the window
            # the best one, so the selected index is the window's start.
            recorded_history(
                vbmc,
                elbo=[float(n - i) for i in range(n)],
                elbo_sd=[0.0] * n,
                r_index=[1.0] * n,
                stable=[False] * n,
            )
            __, __, __, idx_best = vbmc.determine_best_vp(
                frac_back=frac_back, rank_criterion_flag=False
            )
            expected = max(0, (n - 1) - int(np.ceil(n * frac_back)))
            assert idx_best == expected, (n, frac_back, idx_best, expected)


def test_rank_penalty_equals_the_number_of_iterations():
    """The ranking criterion penalizes a non-stable iteration by the number
    of iterations ranked. With four iterations, a stable first iteration
    ranked second on ELCBO and first on reliability totals 4 + 2 + 1 + 1 =
    8, and the last iteration, unstable and ranked first on ELCBO and
    second on reliability, totals 1 + 1 + 2 + 4 = 8; the tie goes to the
    earlier, stable iteration. A penalty one smaller would select the
    unstable one."""
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    recorded_history(
        vbmc,
        elbo=[2.0, 0.0, 1.0, 3.0],
        elbo_sd=[0.0] * 4,
        r_index=[0.0, 3.0, 2.0, 1.0],
        stable=[True, False, False, False],
    )
    __, __, __, idx_best = vbmc.determine_best_vp(rank_criterion_flag=True)
    assert idx_best == 0


def test_equal_elcbo_ranks_the_earlier_iteration_first():
    """MATLAB VBMC ranks the iterations by ELCBO with a descending ``sort``
    (``misc/best_vbmc.m:36``), which is stable: of two iterations with the
    same ELCBO the earlier one gets the better rank. With three iterations,
    none of them stable, the first then totals 3 + 1 + 1 and the second
    2 + 2 + 2, so the first is selected; with the ranks of the tie exchanged
    the second would be."""
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    recorded_history(
        vbmc,
        elbo=[1.0, 1.0, 0.0],
        elbo_sd=[0.1] * 3,
        r_index=[1.0, 2.0, 3.0],
        stable=[False] * 3,
    )
    __, __, __, idx_best = vbmc.determine_best_vp(rank_criterion_flag=True)
    assert idx_best == 0


def test_equal_reliability_ranks_the_earlier_iteration_first():
    """The ranking by reliability index is an ascending ``sort``
    (``misc/best_vbmc.m:40``), stable as well. A run records an infinite
    index for its first two iterations, so they tie for the last two ranks.
    With 22 iterations of which only those two are stable, the first totals
    22 + 1 + 21 + 1 = 45 and the second 21 + 2 + 22 + 1 = 46, and every
    other iteration at least 46 with its penalty of 22; with the ranks of
    the tie exchanged the second would total 45 and be selected."""
    n = 22
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    recorded_history(
        vbmc,
        elbo=[float(n - i) for i in range(n)],
        elbo_sd=[0.0] * n,
        r_index=[np.inf, np.inf] + [float(i) for i in range(2, n)],
        stable=[True, True] + [False] * (n - 2),
    )
    __, __, __, idx_best = vbmc.determine_best_vp(rank_criterion_flag=True)
    assert idx_best == 0


def test_nan_elcbo_ranks_last():
    """A NaN ELCBO ranks last in the ranking criterion. MATLAB VBMC's
    descending ``sort`` (``misc/best_vbmc.m:36``) ranks it first instead,
    which selects the posterior whose ELBO is NaN. With the NaN ranked
    last, the third iteration totals 1 + 1 + 3 + 3 = 8 (position, ELCBO,
    reliability, penalty), the first 3 + 2 + 1 + 3 = 9 and the second
    2 + 3 + 2 + 3 = 10."""
    for scores in (
        dict(elbo=[1.0, np.nan, 2.0], elbo_sd=[0.0] * 3),
        dict(elbo=[1.0, 3.0, 2.0], elbo_sd=[0.0, np.nan, 0.0]),
    ):
        vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
        recorded_history(vbmc, **scores, r_index=[1.0] * 3, stable=[False] * 3)
        __, __, __, idx_best = vbmc.determine_best_vp(rank_criterion_flag=True)
        assert idx_best == 2, scores


def test_nan_reliability_index_ranks_last():
    """A NaN reliability index ranks last, after the infinite index of the
    first two iterations. With five iterations, none of them stable and an
    ELCBO that falls with the iteration, the ranks by position and by ELCBO
    sum to 6 for every iteration, so the reliability rank decides: the NaN
    of the fourth iteration ranks it fifth, and the fifth iteration, ranked
    first, is selected."""
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    recorded_history(
        vbmc,
        elbo=[5.0, 4.0, 3.0, 2.0, 1.0],
        elbo_sd=[0.0] * 5,
        r_index=[np.inf, np.inf, 2.0, np.nan, 1.0],
        stable=[False] * 5,
    )
    __, __, __, idx_best = vbmc.determine_best_vp(rank_criterion_flag=True)
    assert idx_best == 4


def test_look_back_skips_nan_elcbo():
    """Without the ranking criterion, the best ELCBO of the look-back
    window skips a NaN ELCBO, as MATLAB's ``max`` does, also at the start
    of the window."""
    cases = [
        # The last stable iteration opens the window on a NaN ELBO.
        (dict(elbo=[5.0, np.nan, 1.0, 2.0]), [False, True, False, False], 3),
        # A NaN ELBO SD at the start of the window.
        (dict(elbo=[3.0, 1.0], elbo_sd=[np.nan, 0.0]), [False, False], 1),
        # A NaN in the middle of the window.
        (dict(elbo=[0.0, 2.0, np.nan, 1.0]), [True, False, False, False], 1),
    ]
    for scores, stable, expected in cases:
        n = len(stable)
        scores.setdefault("elbo_sd", [0.0] * n)
        vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
        recorded_history(vbmc, **scores, r_index=[1.0] * n, stable=stable)
        __, __, __, idx_best = vbmc.determine_best_vp(
            rank_criterion_flag=False
        )
        assert idx_best == expected, (scores, stable)


def test_every_elcbo_nan_selects_the_last_iteration():
    """If the ELCBO of every candidate iteration is NaN, the selection
    returns the last iteration considered, the posterior the run ended on.
    Without the ranking criterion the candidates are the iterations of the
    look-back window, which here leaves out the first iteration; with it,
    they are every iteration up to the last one considered."""
    for rank_criterion_flag in (False, True):
        for stable, max_idx in (
            ([False, True, False, False], None),
            ([False, True, False, False], 2),
            ([False] * 4, None),
        ):
            vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
            elbo = [np.nan] * 4
            if not rank_criterion_flag:
                elbo[0] = 1.0
            recorded_history(
                vbmc,
                elbo=elbo,
                elbo_sd=[0.0] * 4,
                r_index=[np.inf, np.inf, np.nan, np.nan],
                stable=stable,
            )
            __, __, __, idx_best = vbmc.determine_best_vp(
                max_idx=max_idx, rank_criterion_flag=rank_criterion_flag
            )
            expected = 3 if max_idx is None else max_idx
            assert idx_best == expected, (rank_criterion_flag, stable)


def test_determine_best_vp_defaults_to_the_run_options():
    """Called without selection arguments, ``determine_best_vp`` uses the
    run's options ``best_safe_sd``, ``best_frac_back`` and
    ``rank_criterion``, so that on a finished run it returns the iteration
    the run selected. In both histories below the last iteration is not
    stable."""
    ranked = dict(
        elbo=[2.0, 0.0, 1.0, 3.0],
        elbo_sd=[0.0, 0.0, 0.0, 0.1],
        r_index=[0.0, 3.0, 2.0, 1.0],
        stable=[True, False, False, False],
    )
    unstable = dict(
        elbo=[4.0, 3.0, 2.0, 1.0, 0.0],
        elbo_sd=[0.0] * 5,
        r_index=[1.0] * 5,
        stable=[False] * 5,
    )
    cases = [
        # The ranking criterion, on by default, selects the stable first
        # iteration, as in test_rank_penalty_equals_the_number_of_iterations.
        (ranked, {}, 0),
        # Without it, the look-back since that iteration selects the last
        # one, whose ELCBO is the best,
        (ranked, {"rank_criterion": False}, 3),
        # unless a larger penalty on its uncertainty makes the first one
        # the best.
        (ranked, {"rank_criterion": False, "best_safe_sd": 20}, 0),
        # Without a stable iteration, the look-back window covers the
        # fraction best_frac_back of the iterations.
        (unstable, {"rank_criterion": False}, 2),
        (unstable, {"rank_criterion": False, "best_frac_back": 1.0}, 0),
    ]
    for scores, options, expected in cases:
        vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
        recorded_history(vbmc, **scores)
        __, __, __, idx_default = vbmc.determine_best_vp()
        __, __, __, idx_options = vbmc.determine_best_vp(
            safe_sd=vbmc.options["best_safe_sd"],
            frac_back=vbmc.options["best_frac_back"],
            rank_criterion_flag=vbmc.options["rank_criterion"],
        )
        assert idx_default == idx_options == expected, options

    # An argument given explicitly overrides the option. The defaults of
    # the arguments in release 1.0.4, an ELCBO penalty of 5, a quarter of
    # the iterations and no ranking criterion, select another iteration of
    # the first history than the default options do.
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    recorded_history(vbmc, **ranked)
    assert vbmc.options["rank_criterion"]
    __, __, __, idx_best = vbmc.determine_best_vp(
        safe_sd=5, frac_back=0.25, rank_criterion_flag=False
    )
    assert idx_best == 3
