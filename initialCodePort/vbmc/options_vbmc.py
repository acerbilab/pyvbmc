from collections.abc import MutableMapping
from .default_options_advanced import default_options_advanced
from .default_options_fixed import default_options_fixed


class OptionsVBMC(MutableMapping, dict):
    def __init__(self, D, K, *args, **kwargs):
        # Advanced options (do not modify unless you *know* what you are doing)
        self.update(default_options_advanced)
        # Advanced options for unsupported/untested features (do *not* modify)
        self.update(default_options_fixed)
        self.update(*args, **kwargs)

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __iter__(self):
        yield from sorted(dict.__iter__(self))

    def __len__(self):
        return dict.__len__(self)