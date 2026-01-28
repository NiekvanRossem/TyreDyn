import warnings
import numpy as np
from typing import Union
from tyredyn.types.aliases import SignalLike, AngleUnit

class TyreBase:
    """
    Base template for tyre classes. This class contains the methods that are shared between the subclasses. Do not use
    directly.
    """

    def __init_from_data__(self, params: dict, **settings):
        """
        Unpack and stores the tyre parameter dictionary, reads the user-defined settings, and sets the trigonometry
        functions based on the angle unit used to fit the tyre coefficients.

        Parameters
        ----------
        params : dict
            Dictionary containing TIR file parameters. Separated in the same sections as the TIR file.
        settings
            Optional user settings (see documentation).
        """

        # store units separately in a dictionary
        self._tir_units = {}
        for i, dimension in enumerate(params["UNITS"]):
            self._tir_units[dimension] = params["UNITS"][dimension].replace("'", "")

        # store user settings
        self._use_alpha_star    = settings.get('use_alpha_star', True)
        self._use_gamma_star    = settings.get('use_gamma_star', True)
        self._use_turn_slip     = settings.get('use_turn_slip', False)
        self._use_lmu_star      = settings.get('use_lmu_star', True)
        self._use_mfeval_mode   = settings.get('use_mfeval_mode', False)
        self._check_format      = settings.get('check_format', True) # TODO: change name
        self._check_limits      = settings.get('check_limits', True)
        self._use_model_type    = settings.get('use_model_type', None)

        # unpack parameter dictionary
        self._params_flat = {}
        for sec_data in params.values():
            for k, v in sec_data.items():
                if isinstance(v, str):
                    pass
                else:
                    self._params_flat[k] = v

        # correct trigonometry functions for TIR files fitted in degrees EXPERIMENTAL
        if self._tir_units["ANGLE"] in ["rad", "radian", "radians"]:
            self.sin  = lambda x: np.sin(x)
            self.cos  = lambda x: np.cos(x)
            self.tan  = lambda x: np.tan(x)
            self.asin = lambda x: np.asin(x)
            self.acos = lambda x: np.acos(x)
            self.atan = lambda x: np.atan2(x, 1.0)
        elif self._tir_units["ANGLE"] in ["deg", "degree", "degrees"]:
            warnings.warn("TIR file coefficients fitted in degrees. This is supported but very experimental! Use with "
                          "caution.")
            self.sin  = lambda x: np.sin(np.deg2rad(x))
            self.cos  = lambda x: np.cos(np.deg2rad(x))
            self.tan  = lambda x: np.tan(np.deg2rad(x))
            self.asin = lambda x: np.asin(np.deg2rad(x))
            self.acos = lambda x: np.acos(np.deg2rad(x))
            self.atan = lambda x: np.atan2(np.deg2rad(x), 1.0)

    def __getattr__(self, item):
        """Make the tyre parameters available in the functions."""
        params = object.__getattribute__(self, "_params_flat")
        if item in params:
            return params[item]
        else:
            raise AttributeError(f"{type(self).__name__} has no attribute '{item}'")
