from tyredyn.types.aliases import NumberLike, SignalLike, AngleUnit
import numpy as np

class ProcessInputs:
    """Class containing the methods for pre-processing the input data."""

    def __init__(self, model):
        """Make the properties of the overarching class and other subsystems available."""
        self._model = model

        # helper functions
        self.correction    = model.correction
        self.normalize     = model.normalize
        self.signals       = model.signals
        self.extra_signals = model.extra_signals
        self.data_checks   = model.data_checks

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    def _process_data(
            self,
            *,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            N:    SignalLike,
            P:    SignalLike,
            IA:   SignalLike,
            VX:   SignalLike,
            PHIT: SignalLike,
            angle_unit: AngleUnit,
    ) -> list[SignalLike]:
        """
        Pre-processes the provided input signals, and prepares them for use in the functions.

        The following operations are performed:
          - Sets default values for pressure and speed if not already provided.
          - Estimates the angular speed of the wheel if not already provided.
          - Add low speed correction.
          - Clip ``FZ`` values to avoid negative contact patch force.
          - Compare the angle unit of the TIR file and the input signals, and makes the necessary connections.
          - Checks the format of the input signal: all should be 1D arrays or lists of the same length (or length 1),
            made up of real numbers (optional).
          - Checks and limits the input signals to the min and max values specified in the TIR file (optional).

        Parameters
        ----------
        SA : SignalLike
            Slip angle.
        SL : SignalLike
            Slip ratio.
        FZ : SignalLike
            Vertical force
        N : SignalLike
            Angular wheel speed.
        P : SignalLike
            Tyre pressure.
        IA : SignalLike
            Inclination angle with respect to the ground plane.
        VX : SignalLike
            Contact patch longitudinal speed.
        PHIT : SignalLike
            Turn slip.
        angle_unit : str
            Unit of the signals indicating an angle.

        Returns
        -------
        out: list[SignalLike, AngleUnit]
            The same signals you provided as an input, but corrected, in the same order as shown above.
        """

        # set default values for optional arguments
        P, VX = self.__set_default_values(P, VX)

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            SA, SL, FZ, N, P, IA, VX, PHIT = self.data_checks._format_check([SA, SL, FZ, N, P, IA, VX, PHIT])

        # check if inputs fall within the specified limits
        if self._check_limits:
            self.data_checks._limit_check(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, angle_unit=angle_unit)

        # approximate N if it is not specified
        N = self.extra_signals._find_omega(SL=SL, FZ=FZ, VX=VX, P=P) if N is None else N

        # correct angle if mismatched between input array and TIR file
        [SA, IA, N], angle_unit = self.data_checks._angle_unit_check([SA, IA, N], angle_unit=angle_unit)

        # remove negative FZ values
        FZ = np.maximum(FZ, 0.0)

        # correct turn slip (empirically discovered by Marco Furlan)
        PHIT = PHIT * self.cos(SA) if SA is not None else PHIT

        # correct signals for low speed
        SA, SL, PHIT = self.__low_speed_correction(SA=SA, SL=SL, VX=VX, PHIT=PHIT)

        # low speed correction for slip ratio and turn slip
        #linear_correction = np.abs(VX / self.VXLOW)
        #PHIT = self.signals._correct_signal(PHIT, correction_factor=linear_correction, helper_sig=VX, condition="<", threshold=self.VXLOW)
        #SL   = self.signals._correct_signal(SL,   correction_factor=linear_correction, helper_sig=VX, condition="<", threshold=self.VXLOW)

        # lateral slip speed (2.12)
        #VSY = VX * self.tan(SA)

        # low speed correction for slip angle
        #speed_sum = np.abs(VX) + np.abs(VSY)
        #alpha_correction = speed_sum / self.VXLOW
        #SA = self.signals._correct_signal(SA, correction_factor=alpha_correction, helper_sig=speed_sum, condition="<", threshold=self.VXLOW)

        # flip the sign of the turn slip for negative speeds
        #PHIT = self.signals._flip_negative(PHIT, helper_sig=VX)

        return [SA, SL, FZ, N, P, IA, VX, PHIT, angle_unit]

    #------------------------------------------------------------------------------------------------------------------#
    # INTERNAL FUNCTIONS

    def __set_default_values(self, P, VX):
        """set default values for optional arguments"""

        P  = self.INFLPRES if P is None else P
        VX = self.LONGVL if VX is None else VX
        return [P, VX]

    def __low_speed_correction(self, *, SA, SL, VX, PHIT):
        """
        Corrects the slip angle, slip ratio, and turn slip for low speeds (less than VXLOW) to prevent singularities.
        """

        # low speed correction for slip ratio and turn slip
        linear_correction = np.abs(VX / self.VXLOW)
        PHIT = self.signals._correct_signal(PHIT, correction_factor=linear_correction, helper_sig=VX, condition="<",
                                              threshold=self.VXLOW)
        SL = self.signals._correct_signal(SL, correction_factor=linear_correction, helper_sig=VX, condition="<",
                                            threshold=self.VXLOW)

        # lateral slip speed (2.12 from the 2012 book)
        VSY = VX * self.tan(SA)

        # low speed correction for slip angle (via Marco Furlan)
        speed_sum = np.abs(VX) + np.abs(VSY)
        alpha_correction = speed_sum / self.VXLOW
        SA = self.signals._correct_signal(SA, correction_factor=alpha_correction, helper_sig=speed_sum, condition="<",
                                            threshold=self.VXLOW)

        # flip the sign of the turn slip for negative speeds (via Marco Furlan)
        PHIT = self.signals._flip_negative(PHIT, helper_sig=VX)

        return [SA, SL, PHIT]