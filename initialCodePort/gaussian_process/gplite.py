from abc import ABC, abstractmethod


class GP_Lite(ABC):
    """
    ​Use interface and common GP library for first version of pyVBMC
    """

    @abstractmethod
    def gplite_train(self, gp, Xstar, ystar, s2star, ssflag, nowarpflag):
        """
        docstring
        """
        raise NotImplementedError

    @abstractmethod
    def gplite_predict(self, parameter_list):
        """
        docstring
        """
        raise NotImplementedError