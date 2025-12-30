from src.utils.formatting import SignalLike, AngleUnit
from typing import Literal
import numpy as np

class MomentsMF61:
    """
    Moments module for MF 6.1.
    """

    def __init__(self, model):
        """Make the properties of the overarching ``MF61`` class and other modules available."""
        self._model = model

        # helper functions
        self.correction = model.correction
        self.normalize  = model.normalize
        self.common     = model.common

        # other modules
        self.turn_slip  = model.turn_slip
        self.friction   = model.friction
        self.gradient   = model.gradient
        self.forces     = model.forces
        self.trail      = model.trail

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    # ------------------------------------------------------------------------------------------------------------------#
    # PURE SLIP MOMENTS

    def find_mx_pure(
            self,
            *,
            SA:   SignalLike,
            FZ:   SignalLike,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VCX:  SignalLike = None,
            VS:   SignalLike = 0.0,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:
        """
        Returns the overturning couple for pure slip conditions.

        Parameters
        ----------
        SA : SignalLike
            Slip angle.
        FZ : SignalLike
            Vertical load.
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : SignalLike, optional
            Inclination angle with respect to the ground plane (will default to zero if not specified).
        VCX : SignalLike, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).
        VS : SignalLike, optional
            Slip speed magnitude (will default to zero if not specified).
        PHIT : SignalLike, optional
            Turn slip (will default to zero if not specified).
        angle_unit : string, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        MX : SignalLike
            Overturning couple for pure slip conditions.
        """

        # set default values for optional arguments
        P   = self.INFLPRES if P is None else P
        VCX = self.LONGVL if VCX is None else VCX

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            SA, FZ, P, IA, VCX, VS, PHIT = self._format_check([SA, FZ, P, IA, VCX, VS, PHIT])

        # correct angle if mismatched between input array and TIR file
        [SA, IA], angle_unit = self._angle_unit_check([SA, IA], angle_unit)

        # find side force
        FY = self.forces.find_fy_pure(SA=SA, FZ=FZ, P=P, IA=IA, VCX=VCX, VS=VS, PHIT=PHIT, angle_unit=angle_unit)

        # find overturning moment
        MX = self.__mx_main_routine(FY=FY, FZ=FZ, P=P, IA=IA)
        return MX

    def find_my_pure(
            self,
            *,
            SL:  SignalLike,
            FZ:  SignalLike,
            P:   SignalLike = None,
            IA:  SignalLike = 0.0,
            VC:  SignalLike = None,
            VCX: SignalLike = None,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:
        """
        Returns the rolling resistance couple for pure slip conditions.

        Parameters
        ----------
        VC
        SL : SignalLike
            Slip ratio.
        FZ : SignalLike
            Vertical load.
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : SignalLike, optional
            Inclination angle with respect to the ground plane (will default to zero if not specified).
        VCX : SignalLike, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).
        angle_unit : string, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        MY : SignalLike
            Rolling resistance couple for pure slip conditions.
        """

        # set default values for optional arguments
        P  = self.INFLPRES if P is None else P
        VC = self.LONGVL if VC is None else VC
        VCX = self.LONGVL if VCX is None else VCX

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            SL, FZ, P, IA, VCX = self._format_check([SL, FZ, P, IA, VCX])

        # correct angle if mismatched between input array and TIR file
        IA, angle_unit = self._angle_unit_check(IA, angle_unit)

        # calculate FX
        FX = self.forces.find_fx_pure(SL=SL, FZ=FZ, P=P, IA=IA, VC=VC, VS=VS, PHIT=PHI, angle_unit=angle_unit)

        # find rolling resistance moment
        MY = self.__my_main_routine(FX=FX, FZ=FZ, P=P, IA=IA, VCX=VCX)
        return MY

    def find_mz_pure(
            self,
            *,
            SA:   SignalLike,
            FZ:   SignalLike,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VC:   SignalLike = None,
            VCX:  SignalLike = None,
            VS:   SignalLike = 0.0,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:
        """
        Returns the self-aligning couple for pure slip conditions.

        Parameters
        ----------
        SA : SignalLike
            Slip angle.
        FZ : SignalLike
            Vertical load.
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : SignalLike, optional
            Inclination angle with respect to the ground plane (will default to zero if not specified).
        VC : SignalLike, optional
            Contact patch speed (will default to ``LONGVL`` if not specified).
        VCX : SignalLike, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).
        VS : SignalLike, optional
            Slip speed magnitude (will default to zero if not specified).
        PHIT : SignalLike, optional
            Turn slip (will default to zero if not specified).
        angle_unit : string, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        MZ : SignalLike
            Self-aligning couple for pure slip conditions.
        """

        # set default values for optional arguments
        P   = self.INFLPRES if P is None else P
        VC  = self.LONGVL if VC is None else VC
        VCX = self.LONGVL if VCX is None else VCX

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            SA, FZ, P, IA, VC, VCX, VS, PHIT = self._format_check([SA, FZ, P, IA, VC, VCX, VS, PHIT])

        # correct angle if mismatched between input array and TIR file
        [SA, IA], angle_unit = self._angle_unit_check([SA, IA], angle_unit)

        # turn slip correction
        if self._use_turn_slip:
            PHI = self.correction._find_phi(FZ=FZ, N=N, VC=VC, IA=IA, PHIT=PHIT)
            zeta_0 = 0.0  # (4.83)
            zeta_2 = self.turn_slip._find_zeta_2(SA=SA, FZ=FZ, PHI=PHI)
            zeta_4 = self.turn_slip._find_zeta_4(FZ=FZ, P=P, IA=IA, VCX=VCX, VS=VS, PHI=PHI, zeta_2=zeta_2, angle_unit=angle_unit)
            zeta_6 = self.turn_slip._find_zeta_6(PHI)
            zeta_7 = self.turn_slip._find_zeta_7(SA=SA, SL=0.0, FZ=FZ, P=P, IA=IA, VCX=VCX, VS=VS, PHI=PHI, angle_unit=angle_unit)
            zeta_8 = self.turn_slip._find_zeta_8(FZ=FZ, P=P, IA=IA, VS=VS, PHI=PHI, angle_unit=angle_unit)
        else:
            zeta_0 = self.zeta_default
            zeta_2 = self.zeta_default
            zeta_4 = self.zeta_default
            zeta_6 = self.zeta_default
            zeta_7 = self.zeta_default
            zeta_8 = self.zeta_default

        # pneumatic trail
        t = self.trail.find_trail_pure(SA=SA, FZ=FZ, P=P, IA=IA, VC=VC, VCX=VCX, VS=VS, PHIT=PHIT, angle_unit=angle_unit)

        # find side force
        FY = self.forces.find_fy_pure(SA=SA, FZ=FZ, P=P, IA=IA, VCX=VCX, VS=VS, PHIT=PHIT, angle_unit=angle_unit)

        # residual self-aligning couple (4.E36)
        MZR = self._mz_main_routine(SA=SA, SL=0.0, FZ=FZ, P=P, IA=IA, VC=VC, VCX=VCX, VS=VS, zeta_0=zeta_0,
                                    zeta_2=zeta_2, zeta_4=zeta_4, zeta_6=zeta_6, zeta_7=zeta_7, zeta_8=zeta_8,
                                    combined_slip=False, angle_unit=angle_unit)

        # self-aligning couple due to pneumatic trail (4.E32)
        MZ_prime = - t * FY

        # final self-aligning couple (4.E31)
        MZ = MZ_prime + MZR

        return MZ

    # ------------------------------------------------------------------------------------------------------------------#
    # COMBINED SLIP MOMENTS

    def find_mx_combined(
            self,
            *,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VCX:  SignalLike = None,
            VS:   SignalLike = 0.0,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:
        """
        Returns the overturning couple for combined slip conditions.

        Parameters
        ----------
        SA : SignalLike
            Slip angle.
        SL : SignalLike
            Slip ratio.
        FZ : SignalLike
            Vertical load.
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : SignalLike, optional
            Inclination angle with respect to the ground plane (will default to zero if not specified).
        VCX : SignalLike, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).
        VS : SignalLike, optional
            Slip speed magnitude (will default to zero if not specified).
        PHIT : SignalLike, optional
            Turn slip (will default to zero if not specified).
        angle_unit : string, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        MX : SignalLike
            Overturning couple for combined slip conditions.
        """

        # set default values for optional arguments
        P   = self.INFLPRES if P is None else P
        VCX = self.LONGVL if VCX is None else VCX

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            SA, SL, FZ, P, IA, VCX, VS, PHIT = self._format_check([SA, SL, FZ, P, IA, VCX, VS, PHIT])

        # correct angle if mismatched between input array and TIR file
        [SA, IA], angle_unit = self._angle_unit_check([SA, IA], angle_unit)

        # find side force
        FY = self.forces.find_fy_combined(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VCX=VCX, VS=VS, PHIT=PHIT,
                                          angle_unit=angle_unit)

        # find overturning couple
        MX = self.__mx_main_routine(FY=FY, FZ=FZ, P=P, IA=IA)
        return MX

    def find_my_combined(
            self,
            *,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VC:   SignalLike = None,
            VCX:  SignalLike = None,
            VS:   SignalLike = 0.0,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:
        """
        Returns the rolling resistance couple for combined slip conditions.

        Parameters
        ----------
        SA : SignalLike
            Slip angle.
        SL : SignalLike
            Slip ratio.
        FZ : SignalLike
            Vertical load.
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : SignalLike, optional
            Inclination angle with respect to the ground plane (will default to zero if not specified).
        VC : SignalLike, optional
            Contact patch speed (will default to ``LONGVL`` if not specified).
        VCX : SignalLike, optional
            Wheel center longitudinal speed (will default to ``LONGVL`` if not specified).
        VS : SignalLike, optional
            Contact patch slip speed (will default to zero if not specified).
        PHIT: SignalLike, optional
            Turn slip (will default to zero if not specified).
        angle_unit : string, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        MY : SignalLike
            Rolling resistance couple for combined slip conditions.
        """

        # set default values for optional arguments
        # NOTE: it is assumed that difference between contact patch and wheel center speed is negligible as (eqn 7.4
        # from Pacejka)
        P   = self.INFLPRES if P is None else P
        VCX = self.LONGVL if VCX is None else VCX

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            SA, SL, FZ, P, IA, VCX, VS, PHIT = self._format_check([SA, SL, FZ, P, IA, VCX, VS, PHIT])

        # correct angle if mismatched between input array and TIR file
        [SA, IA], angle_unit = self._angle_unit_check([SA, IA], angle_unit)

        # calculate FX
        FX = self.forces.find_fx_combined(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VC=VC, VCX=VCX, VS=VS, PHIT=PHIT,
                                          angle_unit=angle_unit)

        # find rolling resistance moment
        MY = self.__my_main_routine(FX=FX, FZ=FZ, P=P, IA=IA, VCX=VCX)
        return MY

    def find_mz_combined(
            self,
            *,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VC:   SignalLike = None,
            VCX:  SignalLike = None,
            VS:   SignalLike = 0.0,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:
        """
        Returns the self-aligning couple for combined slip conditions.

        Parameters
        ----------
        SA : SignalLike
            Slip angle.
        SL : SignalLike
            Slip ratio.
        FZ : SignalLike
            Vertical load.
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : SignalLike, optional
            Inclination angle with respect to the ground plane (will default to zero if not specified).
        VC : SignalLike, optional
            Contact patch speed (will default to ``LONGVL`` if not specified).
        VCX : SignalLike, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).
        VS : SignalLike, optional
            Contact patch slip speed (will default to ``LONGVL`` if not specified).
        PHIT : SignalLike, optional
            Turn slip (will default to zero if not specified).
        angle_unit : string, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        MZ : SignalLike
            Self-aligning couple for combined slip conditions.
        """

        # set default values for optional arguments
        P   = self.INFLPRES if P is None else P
        VC  = self.LONGVL if VC is None else VC
        VCX = self.LONGVL if VCX is None else VCX

        # unpack tyre properties
        R0  = self.UNLOADED_RADIUS
        FZ0 = self.FNOMIN

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            SA, SL, FZ, P, IA, VC, VCX, VS, PHIT = self._format_check([SA, SL, FZ, P, IA, VC, VCX, VS, PHIT])

        # correct angle if mismatched between input array and TIR file
        [SA, IA], angle_unit = self._angle_unit_check([SA, IA], angle_unit)

        # turn slip correction
        if self._use_turn_slip:
            PHI = self.correction._find_phi(FZ=FZ, N=N, VC=VC, IA=IA, PHIT=PHIT)
            zeta_0 = 0.0
            zeta_2 = self.turn_slip._find_zeta_2(SA=SA, FZ=FZ, PHI=PHIT)
            zeta_4 = self.turn_slip._find_zeta_4(FZ=FZ, P=P, IA=IA, VCX=VCX, VS=VS, PHI=PHI, zeta_2=zeta_2, angle_unit=angle_unit)
            zeta_6 = self.turn_slip._find_zeta_6(PHIT)
            zeta_7 = self.turn_slip._find_zeta_7(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VCX=VCX, VS=VS, PHI=PHI, angle_unit=angle_unit)
            zeta_8 = self.turn_slip._find_zeta_8(FZ=FZ, P=P, IA=IA, VS=VS, PHI=PHI, angle_unit=angle_unit)
        else:
            zeta_0 = self.zeta_default
            zeta_2 = self.zeta_default
            zeta_4 = self.zeta_default
            zeta_6 = self.zeta_default
            zeta_7 = self.zeta_default
            zeta_8 = self.zeta_default

        # scaled nominal loads
        #FZ0_prime = FZ0 * self.LFZO

        # normalize pressure and load
        dfz = self.normalize._find_dfz(FZ)

        # corrected camber angle
        #gamma_star = self.correction._find_gamma_star(IA)

        # tyre forces
        FX = self.forces.find_fx_combined(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VC=VC, VCX=VCX, VS=VS, PHIT=PHIT, angle_unit=angle_unit)
        FY = self.forces.find_fy_combined(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VCX=VCX, VS=VS, PHIT=PHIT, angle_unit=angle_unit)

        # combined slip scaling factor for side force
        GYK = self.common._find_gyk(SA=SA, SL=SL, FZ=FZ, IA=IA, VCX=VCX)

        # NOTE: in the equation above inclination angle is taken into account to match the TNO solver (via Marco Furlan).

        # pure slip side force without camber or turn slip
        FY0 = self.forces.find_fy_pure(SA=SA, FZ=FZ, P=P, IA=0.0, VCX=VCX, VS=VS, PHIT=0.0, angle_unit=angle_unit)

        # combined slip side force (4.E74)
        FY_prime = FY0 * GYK

        # pneumatic trail
        t = self.trail.find_trail_combined(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VC=VC, VCX=VCX, VS=VS, PHIT=PHIT, angle_unit=angle_unit)

        # pneumatic scrub (A56)
        s = R0 * (self.SSZ1 + self.SSZ2 * (FY / FZ0) + (self.SSZ3 + self.SSZ4 * dfz) * IA) * self.LS

        # NOTE: The paper uses FZ0 in the equation above (A56), instead of FZ0_prime, which the book uses (4.E76, shown
        # below). The equation in the paper matches the TNO solver better, and is thus used (via Marco Furlan). Equation
        # (A56) also uses the uncorrected inclination angle IA instead of gamma_star.
        # s = R0 * (self.SSZ1 + self.SSZ2 * (FY / FZ0_prime) + (self.SSZ3 + self.SSZ4 * dfz) * gamma_star) * self.LS

        # self-aligning couple from side force (4.E72)
        MZ_prime = -t * FY_prime

        # residual self-aligning couple
        MZR = self._mz_main_routine(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VC=VC, VCX=VCX, VS=VS, zeta_0=zeta_0,
                                    zeta_2=zeta_2, zeta_4=zeta_4, zeta_6=zeta_6, zeta_7=zeta_7, zeta_8=zeta_8,
                                    combined_slip=True, angle_unit=angle_unit)

        # final self-aligning couple (4.E71)
        MZ = MZ_prime + MZR + s * FX
        return MZ

    #------------------------------------------------------------------------------------------------------------------#
    # INTERNAL FUNCTIONS

    def __mx_main_routine(
            self,
            *,
            FY: SignalLike,
            FZ: SignalLike,
            P:  SignalLike,
            IA: SignalLike
    ) -> SignalLike:
        """Function containing the main ``MX`` calculation routine. To be used in ``find_mx`` and ``find_mx_pure``."""

        # unpack tyre properties
        R0  = self.UNLOADED_RADIUS
        FZ0 = self.FNOMIN

        # normalize pressure
        dpi = self.normalize._find_dpi(P)

        # define overturning couple parameter sets
        set1 = [self.QSX1, self.QSX2, self.QSX3, self.QSX4, self.QSX5, self.QSX6,
                self.QSX7, self.QSX8, self.QSX9, self.QSX10, self.QSX11]
        set2 = [self.QSX12, self.QSX13, self.QSX14]

        # TODO: check if the book equation is for MF 6.1 specifically. In that case split them up.
        # NOTE: the equation manual for MF 6.2 states that the equation for the overturning couple needs to be split up
        # into two parts: the first using the parameters QSX1 to QSX11, and the second using QSX12 to QSX14. The first
        # set is the general formulation, and the second set is an alternative formulation mainly used for motorcycle
        # tyres. If one set is used, the other should be zero! If for any reason a TIR file is provided with non-zero
        # entries in both sets, a warning will be displayed. The equation used in this case is taken from a draft
        # version of the 2010 paper by Besselink et al. (via Marco Furlan)
        if any(x != 0 for x in set1) and any(y != 0 for y in set2):
            warnings.warn("Cannot have non-zero values for both parameter sets QSX1 to QSX11 and QSX12 to QSX14.")

            # overturning couple (49)
            MX = (R0 * FZ * LMX * (self.QSX1 * self.LVMX - self.QSX2 * IA * (1.0 + self.PPMX1 * dpi)
                                   - self.QSX12 * IA * np.abs(IA) + self.QSX3 * FY / FZ0 + self.QSX4
                                   * np.cos(self.QSX5 * np.atan2((self.QSX6 * FZ / FZ0) ** 2, 1))
                                  * np.sin(self.QSX7 * IA + self.QSX8 * np.atan2(self.QSX9 * FY / FZ0, 1))
                                  + self.QSX10 * np.atan(self.QSX11 * FZ / FZ0) * IA)
                  + R0 * FY * self.LMX * (self.QSX13 + self.QSX14 * np.abs(IA)))

        # NOTE: in the cases where only a single parameter set is used, the equation is taken from the MF 6.2 equation
        # manual, instead of (4.E69) from the 2012 book by Pacejka & Besselink (shown in a comment below), in order to
        # match the TNO solver (via Marco Furlan).
        # MX = R0 * FZ * (self.QSX1 * self.LVMX - self.QSX2 * IA * (1.0 + self.PPMX1 * dpi) + self.QSX3 * FY / FZ0
        #                 + self.QSX4 * np.cos(self.QSX5 * np.atan2(self.QSX6 * FZ / FZ0, 1) ** 2)
        #                 * np.sin(self.QSX7 * IA + self.QSX8 * np.atan2(self.QSX9 * FY / FZ0, 1)) + self.QSX10
        #                 * np.atan2(self.QSX11 * FZ / FZ0, 1) * IA) * self.LMX
        else:

            # overturning couple (MF 6.2 equation manual) -- FZ trig functions do not get corrected to degrees
            MX = (R0 * FZ * self.LMX * (self.QSX1 * self.LVMX - self.QSX2 * IA * (1.0 + self.PPMX1 * dpi)
                                       + self.QSX3 * (FY / FZ0) + self.QSX4 * np.cos(self.QSX5 * np.atan2((self.QSX6 * (FZ / FZ0)) ** 2, 1))
                                       * self.sin(self.QSX7 * IA + self.QSX8 * np.atan2(self.QSX9 * (FY / FZ0), 1))
                                       + self.QSX10 * self.atan(self.QSX11 * (FZ / FZ0)) * IA) + R0 * self.LMX
                  * (FY * (self.QSX13 + self.QSX14 * np.abs(IA)) - FZ * self.QSX12 * IA * np.abs(IA)))

        return MX

    def __my_main_routine(
            self,
            *,
            FX:  SignalLike,
            FZ:  SignalLike,
            P:   SignalLike,
            IA:  SignalLike,
            VCX: SignalLike
    ) -> SignalLike:
        """Function containing the main ``MY`` calculation routine. To be used in ``find_my`` and ``find_my_pure``."""

        # unpack tyre properties
        R0  = self.UNLOADED_RADIUS
        V0  = self.LONGVL
        FZ0 = self.FNOMIN
        P0  = self.NOMPRES

        # rolling resistance moment (A48)
        MY = (-R0 * FZ0 * self.LMY * (self.QSY1 + self.QSY2 * (FX / FZ0) + self.QSY3 * np.abs(VCX / V0)
                                     + self.QSY4 * (VCX / V0) ** 4 + (self.QSY5 + self.QSY6 * (FZ / FZ0)) * IA ** 2)
              * ((FZ / FZ0) ** self.QSY7 * (P / P0) ** self.QSY8))

        # NOTE: equation (4.E70) from the book (shown in a comment below) does not match the TNO solver. Instead
        # equation (A48) from the paper is used (via Marco Furlan).
        # MY = (FZ * R0 * (self.QSY1 + self.QSY2 * (FX / FZ0) + self.QSY3 * np.abs(VCX / V0) + self.QSY4 * (VCX / V0) ** 4
        #                 + (self.QSY5 + self.QSY6 * (FZ / FZ0)) * IA ** 2)
        #       * ((FZ / FZ0) ** self.QSY7 * (P / P0) ** self.QSY8) * self.LMY)

        return MY

    def _mz_main_routine(
            self,
            *,
            SA:     SignalLike,
            SL:     SignalLike,
            FZ:     SignalLike,
            P:      SignalLike,
            IA:     SignalLike,
            VC:     SignalLike,
            VCX:    SignalLike,
            VS:     SignalLike,
            zeta_0: SignalLike,
            zeta_2: SignalLike,
            zeta_4: SignalLike,
            zeta_6: SignalLike,
            zeta_7: SignalLike,
            zeta_8: SignalLike,
            combined_slip: bool = False,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:
        """Function containing the main ``MZ`` calculation routine. Used in ``find_mz`` and ``find_mz_pure``."""

        # unpack tyre properties
        R0 = self.UNLOADED_RADIUS
        V0 = self.LONGVL

        # normalize pressure and load
        dfz = self.normalize._find_dfz(FZ)
        dpi = self.normalize._find_dpi(P)

        # corrected camber angle
        gamma_star = self.correction._find_gamma_star(IA)

        # friction scaling factors
        LMUY_star  = self.correction._find_lmu_star(VS=VS, V0=V0, LMU=self.LMUY)
        LMUY_prime = self.correction._find_lmu_prime(LMUY_star)

        # cornering and camber stiffness
        KYA  = self.gradient.find_cornering_stiffness(FZ=FZ)
        KYCO = self.gradient.find_camber_stiffness(FZ=FZ)

        # corrected cornering stiffness (4.E39)
        KYA_prime = KYA + self.eps_kappa * np.sign(KYA)

        # vertical shift for side force (4.E29)
        S_VY, S_VYg = self.common._find_s_vy(FZ=FZ, dfz=dfz, gamma_star=gamma_star, LMUY_prime=LMUY_prime, zeta_2=zeta_2)

        # horizontal shift (4.E27)
        S_HY = self.common._find_s_hy(dfz=dfz, KYA=KYA, KYCO=KYCO, gamma_star=gamma_star, S_VYg=S_VYg, zeta_0=zeta_0, zeta_4=zeta_4)

        # horizontal shift for residual couple (4.E38)
        S_HF = S_HY + S_VY / KYA_prime

        # corrected slip angles (4.E3, 4.E37)
        alpha_star = self.correction._find_alpha_star(SA=SA, VCX=VCX)
        alpha_r = alpha_star + S_HF

        # correction on the slip angle for combined slip
        if combined_slip:

            # slip stiffness
            KXK = self.gradient.find_slip_stiffness(FZ=FZ)

            # corrected slip angle (A54)
            alpha_r_eq = self.atan(np.sqrt(self.tan(alpha_r) ** 2 + (KXK / KYA_prime) ** 2 * SL ** 2)) * np.sign(alpha_r)

            # NOTE: Equation (4.E78) from the book does not match the TNO solver, thus equation (A54) from the paper is
            # used (via Marco Furlan).
            # alpha_r_eq = np.sqrt(alpha_r ** 2 + (KXK / KYA_prime) ** 2 * SL ** 2) * np.sign(alpha_r)

            alpha_used = alpha_r_eq
        else:
            alpha_used = alpha_r

        # friction scaling factor
        LMUY_star = self.correction._find_lmu_star(VS=VS, V0=V0, LMU=self.LMUY)

        # friction coefficient (4.E23)
        mu_y = self.friction.find_mu_y(FZ=FZ, P=P, IA=IA, VS=VS, angle_unit=angle_unit)

        # peak factor (4.E22)
        DY = self.common._find_dy(mu_y=mu_y, FZ=FZ, zeta_2=zeta_2)

        # cosine term correction factor
        cos_prime_alpha = self.correction._find_cos_prime_alpha(VC=VC, VCX=VCX)

        # shape factor (4.E21)
        CY = self.common._find_cy()

        # stiffness factor (4.E26)
        BY = self.common._find_by(FZ=FZ, KYA=KYA, CY=CY, DY=DY)

        # stiffness factor for the residual couple (4.E45)
        BR = (self.QBZ9 * self.LYKA / LMUY_star + self.QBZ10 * BY * CY) * zeta_6

        # shape factor for the residual couple (4.E46)
        CR = zeta_7

        # peak factor for residual couple (4.E47)
        DR = (FZ * R0 * ((self.QDZ6 + self.QDZ7 * dfz) * self.LRES * zeta_2
                         + ((self.QDZ8 + self.QDZ9 * dfz) * (1.0 + self.PPZ2 * dpi)
                             + (self.QDZ10 + self.QDZ11 * dfz) * np.abs(gamma_star))
                         * gamma_star * self.LKZC * zeta_0) * LMUY_star
              * np.sign(VCX) * cos_prime_alpha + zeta_8 - 1.0)

        # residual self-aligning couple (4.E36)
        MZR = DR * self.cos(CR * self.atan(BR * alpha_used)) * cos_prime_alpha

        return MZR
