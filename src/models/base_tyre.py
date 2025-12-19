from src.utils.misc import allowableData

class TyreBase:
    """
    Base template for tyre classes. This class contains all the methods that need to be defined in the subclasses. Do
    not use this class directly.
    """

    def __init_from_data__(self, params: dict, **settings):

        # store units separately in a dictionary
        self._units = params.get("UNITS")

        # store user settings
        self._use_alpha_star = settings.get('use_alpha_star', True)
        self._use_gamma_star = settings.get('use_gamma_star', True)
        self._use_lmu_star   = settings.get('use_lmu_star', True)
        self._check_format   = settings.get('check_format', True)
        self._check_limits   = settings.get('check_limits', True)

        # unpack the dictionary
        self._params_flat = {}
        for sec_data in params.values():
            for k, v in sec_data.items():
                if isinstance(v, str):
                    pass
                else:
                    self._params_flat[k] = v

    # make the tyre parameters available in the functions
    def __getattr__(self, item):
        try:
            return self._params_flat[item]
        except KeyError:
            raise AttributeError(item)

    def __limit_check(self, SA: allowableData = None, SL: allowableData = None, FZ: allowableData = None,
                      P: allowableData = None, IA: allowableData = None):
        """Performs limit checks on the input signal."""
        
        def main(sig_in, minval, maxval, sig_name: str):
            if any(minval < sig_in < maxval):
                raise Warning(f"{sig_name} exceeds specified limits.")

        # pressure check
        try:
            main(P, self.PRESMIN, self.PRESMAX, "Pressure")
        except KeyError or TypeError:
            pass

        # slip angle check
        try:
            main(SA, self.ALPMIN, self.ALPMAX, "Slip angle")
        except KeyError or TypeError:
            pass

        # slip ratio check
        try:
            main(SL, self.KPUMIN, self.KPUMAX, "Slip ratio")
        except KeyError or TypeError:
            pass

        # inclination angle check
        try:
            main(IA, self.CAMMIN, self.CAMMAX, "Inclination angle")
        except KeyError or TypeError:
            pass

        # vertical load check
        try:
            main(FZ, self.FZMIN, self.FZMAX, "Vertical load")
        except KeyError or TypeError:
            pass

    #------------------------------------------------------------------------------------------------------------------#
    """
    def find_fx_pure(self, **kwargs):
        
        raise NotImplementedError(f'{self.model_type} cannot calculate FX or does not exist!')

    def find_fy_pure(self, **kwargs):
        
        raise NotImplementedError(f'{self.model_type} cannot calculate FY or does not exist!')

    #------------------------------------------------------------------------------------------------------------------#

    def find_fx(self, **kwargs):
        
        raise NotImplementedError(f'{self.model_type} cannot calculate FX or does not exist!')

    def find_fy(self, **kwargs):
        
        raise NotImplementedError(f'{self.model_type} cannot calculate FY or does not exist!')

    def find_fz(self, **kwargs):
        
        raise NotImplementedError(f'{self.model_type} cannot calculate FZ or does not exist!')

    #------------------------------------------------------------------------------------------------------------------#

    def find_mx_pure(self, **kwargs):
        
        raise NotImplementedError(f'{self.model_type} cannot calculate MX or does not exist!')

    def find_my_pure(self, **kwargs):
        
        raise NotImplementedError(f'{self.model_type} cannot calculate MY or does not exist!')

    def find_mz_pure(self, **kwargs):
        
        raise NotImplementedError(f'{self.model_type} cannot calculate MZ or does not exist!')

    #------------------------------------------------------------------------------------------------------------------#

    def find_mx(self, **kwargs):
        
        raise NotImplementedError(f'{self.model_type} cannot calculate MX or does not exist!')

    def find_my(self, **kwargs):
        
        raise NotImplementedError(f'{self.model_type} cannot calculate MY or does not exist!')

    def find_mz(self, **kwargs):
        
        raise NotImplementedError(f'{self.model_type} cannot calculate MZ or does not exist!')

    #------------------------------------------------------------------------------------------------------------------#

    def find_forces(self, **kwargs):
        
        raise NotImplementedError(f'{self.model_type} cannot calculate Forces or does not exist!')

    def find_moments(self, **kwargs):
        
        raise NotImplementedError(f'{self.model_type} cannot calculate Forces or does not exist!')
    """