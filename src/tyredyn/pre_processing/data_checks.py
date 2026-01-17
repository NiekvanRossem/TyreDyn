from tyredyn.types.aliases import AngleUnit, SignalLike
from typing import Union, Literal
import numpy as np
import warnings

class DataChecks:

    def __init__(self, model):
        """Make the properties of the overarching class and other subsystems available."""
        self._model = model

        # helper functions
        self.correction = model.correction
        self.normalize  = model.normalize

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    #------------------------------------------------------------------------------------------------------------------#

    def _angle_unit_check(
            self,
            sig_in: Union[SignalLike, list[SignalLike]],
            angle_unit: Literal["rad", "deg"]
    ):
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

        # TODO: currently these functions only convert the input arrays if self._tir_units["ANGLE"] is degrees. This refers
        #  to the TIR coefficients, but not the input arrays!
        #  what it should be:
        #    CASE 1 -- input array in rad, coefficients in rad
        #      simply evaluate np.sin(angle)
        #    CASE 2 -- input array in deg, coefficients in rad!!!
        #      convert input arrays to rad at the very beginning,
        #      now that the inputs are in rad, evaluate as np.sin(angle)
        #    CASE 3 -- input array in rad, coefficients in deg!!!
        #      convert input array to deg,
        #      now that inputs are in deg, evaluate as np.sin(np.rad2deg(angle))
        #    CASE 4 -- input array in deg, coefficients in deg
        #      simply evaluate as np.sin(np.rad2deg(angle))

        assert angle_unit in ["deg", "rad"], "Please set angle_unit to either 'deg' or 'rad'"

        # if the input arrays are specified in degrees, and the coefficients are fitted in radians
        if angle_unit == "deg" and self._tir_units["ANGLE"] in ["rad", "radian", "radians"]:
            if isinstance(sig_in, list):
                n = sig_in.__len__()
                sig_out = n * [None]
                for i, sig in enumerate(sig_in):
                    sig_out[i] = np.deg2rad(sig)
            else:
                sig_out = np.deg2rad(sig_in)

            # update angle unit to match new state
            #angle_unit = "deg"

        # if the input arrays are specified in radians, and the coefficients are fitted in degrees
        elif angle_unit == "rad" and self._tir_units["ANGLE"] in ["deg", "degree", "degrees"]:
            if isinstance(sig_in, list):
                n = sig_in.__len__()
                sig_out = n * [None]
                for i, sig in enumerate(sig_in):
                    sig_out[i] = np.rad2deg(sig)
            else:
                sig_out = np.rad2deg(sig_in)

            # update angle_unit to match new state
            #angle_unit = "deg"

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
            IA: SignalLike = None,
            angle_unit: AngleUnit):
        """
        Checks if the input signals fall within the limits specified in the TIR file.

        Parameters
        ----------
        angle_unit
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

        # TODO: add fix for possible complex number inputs
        # TODO: add low speed limit
        # TODO: add check that all signals are either length 1 or the same length
        # TODO: limit input signals to values specified in TIR file

        # main function for reformatting
        def main(sig_in, minval, maxval, sig_name: str):
            if sig_in is not None:
                if not isinstance(sig_in, np.ndarray):
                    if isinstance(sig_in, list):
                        sig_in = np.array(sig_in)
                    else:
                        sig_in = np.array([sig_in])
                if any(sig_in < minval) or any(sig_in > maxval):
                    warnings.warn(f"{sig_name} exceeds specified limits.")

        if angle_unit == "deg":
            SA = np.deg2rad(SA)
            IA = np.deg2rad(IA)

        # pressure check
        if P is not None:
            main(P, self.PRESMIN, self.PRESMAX, "Pressure")

        # slip angle check
        if SA is not None:
            main(SA, self.ALPMIN, self.ALPMAX, "Slip angle")

        # slip ratio check
        if SL is not None:
            main(SL, self.KPUMIN, self.KPUMAX, "Slip ratio")

        # inclination angle check
        if IA is not None:
            main(IA, self.CAMMIN, self.CAMMAX, "Inclination angle")

        # vertical load check
        if FZ is not None:
            main(FZ, self.FZMIN, self.FZMAX, "Vertical load")

    @staticmethod
    def _format_check(sig_in: Union[SignalLike, list[SignalLike]]) -> SignalLike:
        """
        Checks the shape of the input tyres_example. Valid input signals are:
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
                if signal is None:
                    continue
                else:
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
            if sig_in is None:
                sig_out = sig_in
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
