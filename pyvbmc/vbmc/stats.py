from collections.abc import MutableMapping


class Stats(MutableMapping, dict):
    """
    This class is responsible for VBMC stats.
    """

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __iter__(self):
        yield from sorted(dict.__iter__(self))

    def __len__(self):
        return dict.__len__(self)

    def __delitem__(self, key):
        del self.__dict__[key]

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
    ):
        pass

    def __str__(self):
        """
        Returns the stats in a format key: value.

        Returns
        -------
        str
            The str to describe an stats object.
        """
        return "".join(["{}: {} \n".format(k, v) for (k, v) in self.items()])
