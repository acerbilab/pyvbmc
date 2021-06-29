from collections.abc import MutableMapping
import numpy as np
import copy


class IterationHistory(MutableMapping, dict):
    """
    This class is responsible for the VBMC iteration history.
    """

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, copy.deepcopy(val))

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __iter__(self):
        yield from sorted(dict.__iter__(self))

    def __len__(self):
        return dict.__len__(self)

    def __delitem__(self, key):
        return dict.__delitem__(self, key)

    def record(self, key: str, value: object, iteration: int):
        """
        Store a value for a key in a given iteration.

        Parameters
        ----------
        key : str
            The key for which the value should be stored.
        value : object
            The value which should be stored.
        iteration : int
            The iteration for which the value should be stored, must be >= 0.

        Raises
        ------
        ValueError
            Raised if the value of the iteration is < 0.
        """
        if iteration < 0:
            raise ValueError("The iteration must be >= 0.")
        if key in self:
            if len(self[key]) <= iteration:
                self._expand_array(key, iteration + 1 - len(self[key]))
        else:
            self[key] = np.full([iteration + 1], None)
        self[key][iteration] = copy.deepcopy(value)

    def _expand_array(self, key: str, resize_amount: int):
        """
        A private method to expand the array for a given key by a resize_amount.
        """
        self[key] = np.append(
            self[key], np.full([resize_amount], None), axis=0
        )

    def record_iteration(
        self,
        optim_state,
        vp,
        elbo,
        elbo_sd,
        varss,
        sKL,
        sKL_true,
        gp,
        Ns_gp,
        pruned,
        timer,
        iteration,
    ):
        """
        Convenience method to record a whole pyvbmc iteration.

        Parameters
        ----------
        optim_state : [type]
            [description]
        vp : [type]
            [description]
        elbo : [type]
            [description]
        elbo_sd : [type]
            [description]
        varss : [type]
            [description]
        sKL : [type]
            [description]
        sKL_true : [type]
            [description]
        gp : [type]
            [description]
        Ns_gp : [type]
            [description]
        pruned : [type]
            [description]
        timer : [type]
            [description]
        iteration : int
            The iteration for which the value should be stored, must be >= 0.
        """

        self.record("N", optim_state.get('N'), iteration)
        self.record("Neff", optim_state.get('Neff'), iteration)
        self.record("funccount", optim_state.get('funccount'), iteration)
        self.record("cachecount", optim_state.get('cachecount'), iteration)
        self.record("warmup", optim_state.get('warmup'), iteration)
        self.record("vp", vp, iteration)
        # self.record("vpK", vp.K, iteration)

        self.record("elbo", elbo, iteration)
        self.record("elbo_sd", elbo_sd, iteration)
        self.record("varss", varss, iteration)
        self.record("sKL", sKL, iteration)
        self.record("sKL_true", sKL_true, iteration)
        self.record("lcbmax", optim_state.get("lcbmax"), iteration)

        # gplite_clean(gp) in original MATLAB implementation
        self.record("gp", gp, iteration)
        self.record("gp_N_samples", Ns_gp, iteration)
        # self.record("gp_hyp_full",  gp.get("hyp_full"), iteration)
        self.record("pruned", pruned, iteration)
        # self.record("gp_noise_hpd", np.sqrt(optim_state.get("sn2hpd")), iteration)
        self.record("gp_samplevar", varss, iteration)
        self.record("timer", timer, iteration)

    def __str__(self):
        """
        Returns the iteartion history in a format key: value.

        Returns
        -------
        str
            The str to describe an IterationHistory object.
        """
        return "".join(["{}: {} \n".format(k, v) for (k, v) in self.items()])
