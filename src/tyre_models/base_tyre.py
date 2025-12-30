import warnings
import numpy as np
from typing import Union, Literal
from src.utils.formatting import SignalLike

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
        self._units = {}
        for i, dimension in enumerate(params["UNITS"]):
            self._units[dimension] = params["UNITS"][dimension].replace("'", "")

        # store user settings
        self._use_alpha_star    = settings.get('use_alpha_star', True)
        self._use_gamma_star    = settings.get('use_gamma_star', True)
        self._use_turn_slip     = settings.get('use_turn_slip', False)
        self._use_lmu_star      = settings.get('use_lmu_star', True)
        self._use_mfeval_mode   = settings.get('use_mfeval_mode', False)
        self._check_format      = settings.get('check_format', True) # TODO: change name
        self._check_limits      = settings.get('check_limits', True)

        # unpack parameter dictionary
        self._params_flat = {}
        for sec_data in params.values():
            for k, v in sec_data.items():
                if isinstance(v, str):
                    pass
                else:
                    self._params_flat[k] = v

        # correct trigonometry functions for TIR files fitted in degrees EXPERIMENTAL
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
        params = object.__getattribute__(self, "_params_flat")
        if item in params:
            return params[item]
        else:
            raise AttributeError(f"{type(self).__name__} has no attribute '{item}'")

    def _angle_unit_check(self, sig_in: Union[SignalLike, list[SignalLike]], angle_unit: Literal["rad", "deg"]):
        """
        Checks for possible mismatches between the angle unit of the input arrays and the TIR file, and corrects them.
        If TIR file was created for degrees, and input signals are specified in radians, the signals are converted to
        degrees. If the TIR file was created for radians, and the input signals are specified in degrees, they are
        converted to radians. In any other case they are directly passed through.

        Parameters
        ----------
        sig_in : Union[SignalLike, list[SignalLike]]
            (List of) signal(s) which may need to be converted from degrees to radians or vice versa.
        angle_unit : Literal["rad", "deg"]
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        sig_out : Union[SignalLike, list[SignalLike]]
            Corrected input signals.
        angle_unit : Literal["rad", "deg"]
            Updated angle unit of the input signals.
        """

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

    def _limit_check(
            self,
            *,
            SA: SignalLike = None,
            SL: SignalLike = None,
            FZ: SignalLike = None,
            P:  SignalLike = None,
            IA: SignalLike = None
    ):
        """
        Checks if the input signals fall within the limits specified in the TIR file.

        Parameters
        ----------
        SA : SignalLike, optional
            Slip angle.
        SL : SignalLike, optional
            Slip ratio.
        FZ : SignalLike, optional
            Vertical load.
        P : SignalLike, optional
            Tyre pressure.
        IA : SignalLike, optional
            Inclination angle with respect to the ground plane.
        """

        def main(sig_in, minval, maxval, sig_name: str):
            if sig_in is not None:
                if not isinstance(sig_in, np.ndarray):
                    if isinstance(sig_in, list):
                        sig_in = np.array(sig_in)
                    else:
                        sig_in = np.array([sig_in])
                if any(sig_in < minval) or any(sig_in > maxval):
                    warnings.warn(f"{sig_name} exceeds specified limits.")

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

    @staticmethod
    def _format_check(sig_in: Union[SignalLike, list[SignalLike]]) -> SignalLike:
        """
        Checks the shape of the input data. Valid input signals are:
           - ``int``
           - ``float``
           - ``list`` of either ``int`` or ``float``
           - ``np.ndarray`` (must have shape ``(n,)`` to enforce element-wise algebra)

        If an input signal is a NumPy array with the shape ``(n,1)``, this function will flatten them to ``(n,)``.
        Higher order arrays will generate an error.

        Parameters
        ----------
        sig_in : Union[SignalLike, list[SignalLike]]
            Input signals whose format needs to be checked.

        Returns
        -------
        sig_out : Union[SignalLike, list[SignalLike]]
            Flattened input signals.
        """

        # if a list of channels is passed
        if isinstance(sig_in, list):
            sig_out = len(sig_in) * [None]
            for i, signal in enumerate(sig_in):
                if isinstance(signal, np.ndarray):
                    if signal.ndim > 1:
                        assert signal.shape[1] == 1, "Please input a 1D array."
                        sig_out[i] = signal.flatten()
                    else:
                        sig_out[i] = signal
                else:
                    sig_out[i] = signal

        # if just a single channel is passed
        else:
            if isinstance(sig_in, np.ndarray):
                if sig_in.ndim > 1:
                    assert sig_in.shape[1] == 1, "Please input a 1D array."
                    sig_out = sig_in.flatten()
                else:
                    sig_out = sig_in
            else:
                sig_out = sig_in

        return sig_out
