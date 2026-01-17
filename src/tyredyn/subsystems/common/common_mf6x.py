from tyredyn.types.aliases import SignalLike, AngleUnit, NumberLike
from typing import Union, TypeAlias, Literal
import numpy as np

class CommonMF6x:
    """
    Module containing functions used in multiple other subsystems of MF-Tyre 6.1 and MF-Tyre 6.2.
    """

    def __init__(self, model):
        """Make the properties of the overarching class and other subsystems available."""
        self._model = model

        # helper functions
        self.correction          = model.correction
        self.normalize           = model.normalize
        self.signals             = model.signals
        self.low_speed_reduction = model.low_speed_reduction

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    """
    def _find_omega(
            self,
            SL: SignalLike,
            FZ: SignalLike,
            VX: SignalLike,
            P:  SignalLike,
    ) -> SignalLike:
        \"""Returns the angular speed of the wheel based on the longitudinal velocity and the slip ratio.\"""

        # unpack tyre properties
        FZ0 = self.FNOMIN
        R0  = self.UNLOADED_RADIUS
        V0  = self.LONGVL

        # first guess of angular speed based on unloaded radius
        N = VX / R0
        err = 1
        counter = 0

        # converge over N TODO: add convergence flags and filter for arrays
        while np.max(err) > 1e-9: # TODO: make adjustable setting
            counter += 1

            # free rolling radius
            R_omega = self._find_free_radius(N=N, R0=R0, V0=V0)

            # effective radius
            RE = self._find_effective_radius(FZ=FZ, P=P, R_omega=R_omega, FZ0=FZ0)

            # find new angular speed (2.5)
            N_old = N
            N = ((1.0 + SL) * VX) / RE
            err = np.abs(N - N_old)

            if counter > 100:  # TODO: make adjustable setting
                warnings.warn(f"No estimate for the angular speed found! Final error is {err:.3e}")
                break

        return N
    """

    """
    def _angle_unit_check(self, sig_in: Union[SignalLike, list[SignalLike]], angle_unit: Literal["rad", "deg"]):
        \"""
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
        \"""

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
    """
    """
    def _preprocess_data(
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
        \"""
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
        \"""

        # set default values for optional arguments
        P  = self.INFLPRES if P is None else P
        VX = self.LONGVL if VX is None else VX

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            SA, SL, FZ, N, P, IA, VX, PHIT = self._format_check([SA, SL, FZ, N, P, IA, VX, PHIT])

        # check if inputs fall within the specified limits
        if self._check_limits:
            self._limit_check(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, angle_unit=angle_unit)

        # approximate N if it is not specified
        N = self._find_omega(SL=SL, FZ=FZ, VX=VX, P=P) if N is None else N

        # correct angle if mismatched between input array and TIR file
        assert angle_unit in ["deg","rad"], "Please set angle_unit to either 'deg' or 'rad'"
        [SA, IA, N], angle_unit = self._angle_unit_check([SA, IA, N], angle_unit=angle_unit)

        # remove negative FZ values
        FZ = np.maximum(FZ, 0.0)

        # correct turn slip (empirically discovered by Marco Furlan)
        PHIT = PHIT * self.cos(SA) if SA is not None else PHIT

        # low speed correction for slip ratio and turn slip
        linear_correction = np.abs(VX / self.VXLOW)
        PHIT = self.signals._correct_signal(PHIT, correction_factor=linear_correction, helper_sig=VX, condition="<", threshold=self.VXLOW)
        SL   = self.signals._correct_signal(SL,   correction_factor=linear_correction, helper_sig=VX, condition="<", threshold=self.VXLOW)

        # lateral slip speed (2.12)
        VSY = VX * self.tan(SA)

        # low speed correction for slip angle
        speed_sum = np.abs(VX) + np.abs(VSY)
        alpha_correction = speed_sum / self.VXLOW
        SA = self.signals._correct_signal(SA, correction_factor=alpha_correction, helper_sig=speed_sum, condition="<", threshold=self.VXLOW)

        # flip the sign of the turn slip for negative speeds
        PHIT = self.signals._flip_negative(PHIT, helper_sig=VX)

        return [SA, SL, FZ, N, P, IA, VX, PHIT, angle_unit]
    """

    #------------------------------------------------------------------------------------------------------------------#
    # COMMONLY USED PARAMETERS

    def _find_by(
            self,
            *,
            FZ:  SignalLike,
            KYA: SignalLike,
            CY:  SignalLike,
            DY:  SignalLike
    ) -> SignalLike:
        """Finds the stiffness factor for the side force. Used in ``ForcesMF6x`` and ``MomentsMF6x``."""

        # side force stiffness factor (4.E26)
        eps_y = self._find_eps_y(FZ)
        BY = KYA / (CY * DY + eps_y)
        return BY

    def _find_cy(self) -> SignalLike:
        """Finds the shape factor for the side force. Used in ``ForcesMF6x`` and ``MomentsMF6x``."""

        # (4.E21)
        CY = self.PCY1 * self.LCY
        return CY

    def _find_dt0(
            self,
            *,
            FZ:         SignalLike,
            dfz:        SignalLike,
            dpi:        SignalLike,
            VCX:        SignalLike,
            FZ0_prime:  SignalLike,
            R0:         NumberLike
    ) -> SignalLike:
        """Finds the static peak factor."""

        # (4.E42)
        DT0 = FZ * (R0 / FZ0_prime) * (self.QDZ1 + self.QDZ2 * dfz) * (1.0 - self.PPZ1 * dpi) * self.LTR * np.sign(VCX)
        return DT0

    @staticmethod
    def _find_dy(
            *,
            mu_y:   SignalLike,
            FZ:     SignalLike,
            zeta_2: SignalLike
    ) -> SignalLike:
        """Finds the peak factor for the side force. Used in ``ForcesMF6x`` and ``MomentsMF6x``."""

        # (4.E22)
        DY = mu_y * FZ * zeta_2
        return DY

    def _find_eps_y(
            self,
            FZ: SignalLike
    ) -> SignalLike:
        """Difference between camber and turn slip response. Used internally and in ``TurnSlipMF6x``."""

        if self._use_turn_slip:

            # _normalize load
            dfz = self.normalize._find_dfz(FZ)

            # difference between camber and turn slip response (4.90)
            eps_y = self.PECP1 * (1.0 + self.PECP2 * dfz)

        else:
            eps_y = 1e-6

        return eps_y

    def _find_gyk(
            self,
            *,
            SA:  SignalLike,
            SL:  SignalLike,
            FZ:  SignalLike,
            IA:  SignalLike,
            VCX: SignalLike
    ) -> SignalLike:
        """Returns the side force scaling factor for combined slip conditions."""

        # corrected slip angle (4.E53)
        alpha_star = self.correction._find_alpha_star(SA=SA, VCX=VCX)

        # corrected camber angle (4.E4)
        gamma_star = self.correction._find_gamma_star(IA)

        # _normalize load
        dfz = self.normalize._find_dfz(FZ)

        # stiffness factor (4.E62)
        BYK = (self.RBY1 + self.RBY4 * gamma_star ** 2) * self.cos(self.atan(self.RBY2 * (alpha_star - self.RBY3))) * self.LYKA

        # shape factor (4.E63)
        CYK = self.RCY1

        # curvature factor (4.E64)
        EYK = self.REY1 + self.REY2 * dfz

        # horizontal shift (4.E65)
        S_HYK = self.RHY1 + self.RHY2 * dfz

        # corrected slip ratio (4.E61)
        kappa_s = SL + S_HYK

        # static correction (4.E60) -- slip ratio trig functions do not get corrected to degrees
        GYKO = np.cos(CYK * np.atan2(BYK * S_HYK - EYK * (BYK * S_HYK - np.atan2(BYK * S_HYK, 1)), 1))

        # force correction (4.E59) -- slip ratio trig functions do not get corrected to degrees
        GYK = np.cos(CYK * np.atan2(BYK * kappa_s - EYK * (BYK * kappa_s - np.atan2(BYK * kappa_s, 1)), 1)) / GYKO
        return GYK

    def _find_s_hy(
            self,
            *,
            VX:         SignalLike,
            dfz:        SignalLike,
            KYA:        SignalLike,
            KYCO:       SignalLike,
            gamma_star: SignalLike,
            S_VYg:      SignalLike,
            zeta_0:     SignalLike,
            zeta_4:     SignalLike,
    ) -> SignalLike:
        """Finds the horizontal shift for the side force. Used in ``ForcesMF6x`` and ``MomentsMF6x``."""

        # horizontal shift for side force (4.E27)
        S_HY = ((self.PHY1 + self.PHY2 * dfz) * self.LHY + (KYCO * gamma_star - S_VYg)
                / (KYA + self._eps_kappa) * zeta_0 + zeta_4 - 1.0)

        # low speed correction
        smooth_reduction = self.low_speed_reduction._find_smooth_reduction(VX)
        S_HY = self.signals._correct_signal(S_HY, correction_factor=smooth_reduction, helper_sig=np.abs(VX), threshold=self.VXLOW, condition="<")

        return S_HY

    def _find_s_vy(
            self,
            *,
            FZ:         SignalLike,
            VX:         SignalLike,
            dfz:        SignalLike,
            gamma_star: SignalLike,
            LMUY_prime: SignalLike,
            zeta_2:     SignalLike
    ) -> SignalLike:
        """Finds the vertical shifts for the side force. Used in ``ForcesMF6x``, ``MomentsMF6x``, and ``TurnSlipMF6x``."""

        # vertical shift due to camber (4.E28)
        S_VYg = FZ * (self.PVY3 + self.PVY4 * dfz) * gamma_star * self.LKYC * LMUY_prime * zeta_2

        # total vertical shift (4.E29)
        S_VY = FZ * (self.PVY1 + self.PVY2 * dfz) * self.LVY * LMUY_prime * zeta_2 + S_VYg

        # low speed correction
        smooth_reduction = self.low_speed_reduction._find_smooth_reduction(VX)
        S_VY = self.signals._correct_signal(S_VY, correction_factor=smooth_reduction, helper_sig=np.abs(VX), threshold=self.VXLOW, condition="<")

        return S_VY, S_VYg

    #------------------------------------------------------------------------------------------------------------------#
    # RADII

    """
    def _find_free_radius(
            self,
            *,
            N: SignalLike,
            R0: NumberLike,
            V0: NumberLike
    ) -> SignalLike:

        # free rolling radius (A3.1)
        R_omega = R0 * (self.Q_RE0 + self.Q_V1 * (R0 * N / V0) ** 2)
        return R_omega

    def _find_effective_radius(
            self,
            *,
            FZ: SignalLike,
            P:  SignalLike,
            R_omega: SignalLike,
            FZ0: NumberLike
    ) -> SignalLike:

        # vertical stiffness
        CZ = self.stiffness._find_vertical_stiffness(P)

        # effective radius (A3.6) -- FZ trig functions do not get corrected to degrees
        RE = R_omega - FZ0 / CZ * (self.FREFF * FZ / FZ0 + self.DREFF * np.atan2(self.BREFF * FZ / FZ0, 1))
        return RE
    """