import numpy as np
from typing import Union, Literal
from src.utils.misc import allowableData

class TyreBase:
    """
    Base template for tyre classes. This class contains all the methods that need to be defined in the subclasses. Do
    not use this class directly.
    """

    def __init_from_data__(self, params: dict, **settings):

        # store units separately in a dictionary
        self._units = {}
        for i, dimension in enumerate(params["UNITS"]):
            self._units[dimension] = params["UNITS"][dimension].replace("'", "")

        # store user settings
        self._use_alpha_star = settings.get('use_alpha_star', True)
        self._use_gamma_star = settings.get('use_gamma_star', True)
        self._use_lmu_star   = settings.get('use_lmu_star', True)
        self._check_format   = settings.get('check_format', True)
        self._check_limits   = settings.get('check_limits', True)

        # unpack parameter dictionary
        self._params_flat = {}
        for sec_data in params.values():
            for k, v in sec_data.items():
                if isinstance(v, str):
                    pass
                else:
                    self._params_flat[k] = v


        # correct trigonometry functions for TIR files fitted in degrees. TODO: implement
        if self._units["ANGLE"] in ["rad", "radian", "radians"]:
            self.sin = lambda x: np.sin(x)
            self.cos = lambda x: np.cos(x)
            self.tan = lambda x: np.tan(x)
            self.asin = lambda x: np.asin(x)
            self.acos = lambda x: np.acos(x)
            self.atan = lambda x: np.atan2(x, 1.0)
        elif self._units["ANGLE"] in ["deg", "degree", "degrees"]:
            self.sin  = lambda x: np.sin(np.deg2rad(x))
            self.cos  = lambda x: np.cos(np.deg2rad(x))
            self.tan  = lambda x: np.tan(np.deg2rad(x))
            self.asin = lambda x: np.asin(np.deg2rad(x))
            self.acos = lambda x: np.acos(np.deg2rad(x))
            self.atan = lambda x: np.atan2(np.deg2rad(x), 1)

    def __getattr__(self, item):
        """Make the tyre parameters available in the functions."""
        try:
            return self._params_flat[item]
        except KeyError:
            raise AttributeError(item)

    def _angle_unit_check(self, sig_in: Union[allowableData, list[allowableData]], angle_unit: Literal["rad", "deg"]):
        """Checks for possible mismatches between the angle unit of the input arrays and the TIR file, and corrects them."""

        # if the input arrays are specified in degrees, and the coefficients are fitted in radians
        if angle_unit == "deg" and self._units["ANGLE"] in ["rad", "radian", "radians"]:
            if isinstance(sig_in, list):
                n = sig_in.__len__()
                sig_out = n * [None]
                for i, sig in enumerate(sig_in):
                    sig_out[i] = np.deg2rad(sig)
            else:
                sig_out = np.deg2rad(sig_in)

            # update angle unit to match new state
            angle_unit = "rad"

        # if the input arrays are specified in radians, and the coefficients are fitted in degrees
        elif angle_unit == "rad" and self._units["ANGLE"] in ["deg", "degree", "degrees"]:
            if isinstance(sig_in, list):
                n = sig_in.__len__()
                sig_out = n * [None]
                for i, sig in enumerate(sig_in):
                    sig_out[i] = np.rad2deg(sig)
            else:
                sig_out = np.rad2deg(sig_in)

            # update angle_unit to match new state
            angle_unit = "deg"

        else:
            sig_out = sig_in

        return sig_out, angle_unit

    def _limit_check(self, SA: allowableData = None, SL: allowableData = None, FZ: allowableData = None,
                      P: allowableData = None, IA: allowableData = None):
        """Performs limit checks on the input signal."""
        
        def main(sig_in, minval, maxval, sig_name: str):
            if any(minval < sig_in < maxval): # TODO: make sure sig_in is always np.ndarray
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
