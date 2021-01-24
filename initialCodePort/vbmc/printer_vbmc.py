class Printer_VBMC(object):
    """
    Prints the status messages of the VBMC algorithm
    """

    def __init__(self, verbose_level):
        self.verbose_level = verbose_level

    def print_final_message(self, exitflag):
        #Print final message
        if verbose_level > 1:
            #fprintf('\n%s\n', msg);    
            #fprintf('Estimated ELBO: %.3f +/- %.3f.\n', elbo, elbo_sd);
            #if exitflag < 1
            #    fprintf('Caution: Returned variational solution may have not converged.\n');
            #fprintf('\n');
            print('final message')
