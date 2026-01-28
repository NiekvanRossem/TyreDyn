from tyredyn.types.aliases import SignalLike, AngleUnit
from tyredyn.infrastructure.subsystem_base import SubSystemBase
from typing import Literal
import numpy as np

class MomentsMF6x(SubSystemBase):
    """
    Moments module for the MF-Tyre 6.1 and MF-Tyre 6.2 models.
    """

    def _connect(self, model):

        # helper functions
        self.correction = model.correction
        self.normalize  = model.normalize
        self.common     = model.common
        self.signals    = model.signals

        # other subsystems
        self.turn_slip  = model.turn_slip
        self.friction   = model.friction
        self.gradient   = model.gradient
        self.forces     = model.forces
        self.trail      = model.trail

    # ------------------------------------------------------------------------------------------------------------------#
    # PURE SLIP MOMENTS

    def _find_mx_pure(
            self,
            *,
            SA:   SignalLike,
            FZ:   SignalLike,
            N:    SignalLike,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
            PHIT: SignalLike = 0.0
    ) -> SignalLike:
        """
        Returns the overturning couple for pure slip conditions.

        Parameters
        ----------
        SA : SignalLike
            Slip angle.
        FZ : SignalLike
            Vertical load.
        N : SignalLike, optional
            Angular speed of the wheel (will be calculated from ``VX`` and ``SL`` if not specified).
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : SignalLike, optional
            Inclination angle with respect to the ground plane (will default to ``0.0`` if not specified).
        VX : SignalLike, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).
        PHIT : SignalLike, optional
            Turn slip (will default to ``0.0`` if not specified).

        Returns
        -------
        MX : SignalLike
            Overturning couple for pure slip conditions.
        """

        # find side force
        FY0 = self.forces._find_fy_pure(SA=SA, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

        # find overturning moment
        MX = self.__mx_main_routine(FY=FY0, FZ=FZ, P=P, IA=IA)
        return MX

    def _find_my_pure(
            self,
            *,
            SL:  SignalLike,
            FZ:  SignalLike,
            P:   SignalLike = None,
            IA:  SignalLike = 0.0,
            VX:  SignalLike = None
    ) -> SignalLike:
        """
        Returns the rolling resistance couple for pure slip conditions.

        Parameters
        ----------
        SL : SignalLike
            Slip ratio.
        FZ : SignalLike
            Vertical load.
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : SignalLike, optional
            Inclination angle with respect to the ground plane (will default to ``0.0`` if not specified).
        VX : SignalLike, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).

        Returns
        -------
        MY : SignalLike
            Rolling resistance couple for pure slip conditions.
        """

        # find other velocity components
        VCX = VX

        # calculate FX0
        FX0 = self.forces._find_fx_pure(SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHI)

        # find rolling resistance moment
        MY = self.__my_main_routine(SL=SL, FX=FX0, FZ=FZ, P=P, IA=IA, VCX=VCX)
        return MY

    def _find_mz_pure(
            self,
            *,
            SA:   SignalLike,
            FZ:   SignalLike,
            N:    SignalLike,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
            PHIT: SignalLike = 0.0
    ) -> SignalLike:
        """
        Returns the self-aligning couple for pure slip conditions.

        Parameters
        ----------
        SA : SignalLike
            Slip angle.
        FZ : SignalLike
            Vertical load.
        N : SignalLike, optional
            Angular speed of the wheel (will be calculated from ``VX`` and ``SL`` if not specified).
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : SignalLike, optional
            Inclination angle with respect to the ground plane (will default to ``0.0`` if not specified).
        VX : SignalLike, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).
        PHIT : SignalLike, optional
            Turn slip (will default to ``0.0`` if not specified).

        Returns
        -------
        MZ : SignalLike
            Self-aligning couple for pure slip conditions.
        """

        # find other velocity components
        VS, VC = self.extra_signals._find_speeds(SA=SA, SL=0.0, VX=VX)
        VCX = VX

        # turn slip correction
        if self._use_turn_slip:
            PHI = self.extra_signals._find_phi(FZ=FZ, N=N, VC=VC, IA=IA, PHIT=PHIT)
            zeta_0 = 0.0  # (4.83)
            zeta_2 = self.turn_slip._find_zeta_2(SA=SA, FZ=FZ, PHI=PHI)
            zeta_4 = self.turn_slip._find_zeta_4(SA=SA, SL=0.0, FZ=FZ, N=N, P=P, IA=IA, VCX=VCX, VS=VS, PHI=PHI, zeta_2=zeta_2)
            zeta_6 = self.turn_slip._find_zeta_6(PHI)
            zeta_7 = self.turn_slip._find_zeta_7(SA=SA, SL=0.0, FZ=FZ, P=P, IA=IA, VX=VX, VCX=VCX, PHI=PHI, PHIT=PHIT)
            zeta_8 = self.turn_slip._find_zeta_8(SA=SA, SL=0.0, FZ=FZ, P=P, IA=IA, VX=VX, PHIT=PHIT)
        else:
            zeta_0 = self.zeta_default
            zeta_2 = self.zeta_default
            zeta_4 = self.zeta_default
            zeta_6 = self.zeta_default
            zeta_7 = self.zeta_default
            zeta_8 = self.zeta_default

        # pneumatic trail
        t = self.trail.find_trail_pure(SA=SA, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

        # find side force
        FY = self.forces._find_fy_pure(SA=SA, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

        # residual self-aligning couple (4.E36)
        MZR = self._mz_main_routine(SA=SA, SL=0.0, FZ=FZ, P=P, IA=IA, VX=VX, VC=VC, VCX=VCX, VS=VS, N=N, zeta_0=zeta_0,
                                    zeta_2=zeta_2, zeta_4=zeta_4, zeta_6=zeta_6, zeta_7=zeta_7, zeta_8=zeta_8,
                                    combined_slip=False)

        # self-aligning couple due to pneumatic trail (4.E32)
        MZ_prime = - t * FY

        # final self-aligning couple (4.E31)
        MZ = MZ_prime + MZR

        return MZ

    # ------------------------------------------------------------------------------------------------------------------#
    # COMBINED SLIP MOMENTS

    def _find_mx_combined(
            self,
            *,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            N:    SignalLike,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
            PHIT: SignalLike = 0.0
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
        N : SignalLike, optional
            Angular speed of the wheel (will be calculated from ``VX`` and ``SL`` if not specified).
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : SignalLike, optional
            Inclination angle with respect to the ground plane (will default to ``0.0`` if not specified).
        VX : SignalLike, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).
        PHIT : SignalLike, optional
            Turn slip (will default to ``0.0`` if not specified).

        Returns
        -------
        MX : SignalLike
            Overturning couple for combined slip conditions.
        """

        # find side force
        FY = self.forces._find_fy_combined(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

        # find overturning couple
        MX = self.__mx_main_routine(FY=FY, FZ=FZ, P=P, IA=IA)
        return MX

    def _find_my_combined(
            self,
            *,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            N:    SignalLike,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
            PHIT: SignalLike = 0.0
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
        N : SignalLike, optional
            Angular speed of the wheel (will be calculated from ``VX`` and ``SL`` if not specified).
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : SignalLike, optional
            Inclination angle with respect to the ground plane (will default to ``0.0`` if not specified).
        VX : SignalLike, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).
        PHIT : SignalLike, optional
            Turn slip (will default to ``0.0`` if not specified).

        Returns
        -------
        MY : SignalLike
            Rolling resistance couple for combined slip conditions.
        """

        # find other velocity components
        VCX = VX

        # calculate FX
        FX = self.forces._find_fx_combined(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

        # find rolling resistance moment
        MY = self.__my_main_routine(SL=SL, FX=FX, FZ=FZ, P=P, IA=IA, VCX=VCX)
        return MY

    def _find_mz_combined(
            self,
            *,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            N:    SignalLike,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
            PHIT: SignalLike = 0.0
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
        N : SignalLike, optional
            Angular speed of the wheel (will be calculated from ``VX`` and ``SL`` if not specified).
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : SignalLike, optional
            Inclination angle with respect to the ground plane (will default to ``0.0`` if not specified).
        VX : SignalLike, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).
        PHIT : SignalLike, optional
            Turn slip (will default to ``0.0`` if not specified).

        Returns
        -------
        MZ : SignalLike
            Self-aligning couple for combined slip conditions.
        """

        # find other velocity components
        VS, VC = self.extra_signals._find_speeds(SA=SA, SL=SL, VX=VX)
        VCX = VX

        # unpack tyre properties
        R0  = self.UNLOADED_RADIUS
        FZ0 = self.FNOMIN

        # turn slip correction
        if self._use_turn_slip:
            PHI = self.extra_signals._find_phi(FZ=FZ, N=N, VC=VC, IA=IA, PHIT=PHIT)
            zeta_0 = 0.0
            zeta_2 = self.turn_slip._find_zeta_2(SA=SA, FZ=FZ, PHI=PHIT)
            zeta_4 = self.turn_slip._find_zeta_4(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VCX=VCX, VS=VS, PHI=PHI,
                                                 zeta_2=zeta_2)
            zeta_6 = self.turn_slip._find_zeta_6(PHIT)
            zeta_7 = self.turn_slip._find_zeta_7(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VX=VX, VCX=VCX, PHI=PHI, PHIT=PHIT)
            zeta_8 = self.turn_slip._find_zeta_8(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VX=VX, PHIT=PHIT)
        else:
            zeta_0 = self.zeta_default
            zeta_2 = self.zeta_default
            zeta_4 = self.zeta_default
            zeta_6 = self.zeta_default
            zeta_7 = self.zeta_default
            zeta_8 = self.zeta_default

        # _normalize pressure and load
        dfz = self.normalize._find_dfz(FZ)

        # tyre forces
        FX = self.forces._find_fx_combined(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)
        FY = self.forces._find_fy_combined(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

        # combined slip scaling factor for side force
        # NOTE: in the equation below inclination angle is taken into account to match the TNO solver (via Marco Furlan).
        GYK = self.common._find_gyk(SA=SA, SL=SL, FZ=FZ, IA=IA, VCX=VCX)

        # pure slip side force without camber or turn slip
        FY0 = self.forces._find_fy_pure(SA=SA, FZ=FZ, N=N, P=P, IA=0.0, VX=VX, PHIT=0.0)

        # combined slip side force (4.E74)
        FY_prime = FY0 * GYK

        # pneumatic trail
        t = self.trail.find_trail_combined(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

        # pneumatic scrub (A56)
        s = R0 * (self.SSZ1 + self.SSZ2 * (FY / FZ0) + (self.SSZ3 + self.SSZ4 * dfz) * IA) * self.LS

        # NOTE: The paper uses FZ0 in the equation above (A56), instead of FZ0_prime, which the book uses (4.E76, shown
        # below). The equation in the paper matches the TNO solver better, and is thus used (via Marco Furlan). Equation
        # (A56) also uses the uncorrected inclination angle IA instead of gamma_star.
        # s = R0 * (self.SSZ1 + self.SSZ2 * (FY / FZ0_prime) + (self.SSZ3 + self.SSZ4 * dfz) * gamma_star) * self.LS

        # self-aligning couple from side force (4.E72)
        MZ_prime = -t * FY_prime

        # residual self-aligning couple
        MZR = self._mz_main_routine(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VX=VX, VC=VC, VCX=VCX, VS=VS, N=N, zeta_0=zeta_0,
                                    zeta_2=zeta_2, zeta_4=zeta_4, zeta_6=zeta_6, zeta_7=zeta_7, zeta_8=zeta_8,
                                    combined_slip=True)

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

        # _normalize pressure
        dpi = self.normalize._find_dpi(P)

        # define overturning couple parameter sets
        set1 = [self.QSX1, self.QSX2, self.QSX3, self.QSX4, self.QSX5, self.QSX6,
                self.QSX7, self.QSX8, self.QSX9, self.QSX10, self.QSX11]
        set2 = [self.QSX12, self.QSX13, self.QSX14]

        # NOTE: the equation manual for MF-Tyre 6.2 states that the equation for the overturning couple needs to be split up
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

        # NOTE: in the cases where only a single parameter set is used, the equation is taken from the MF-Tyre 6.2 equation
        # manual, instead of (4.E69) from the 2012 book by Pacejka & Besselink (shown in a comment below), in order to
        # match the TNO solver (via Marco Furlan).
        # MX = R0 * FZ * (self.QSX1 * self.LVMX - self.QSX2 * IA * (1.0 + self.PPMX1 * dpi) + self.QSX3 * FY / FZ0
        #                 + self.QSX4 * np.cos(self.QSX5 * np.atan2(self.QSX6 * FZ / FZ0, 1) ** 2)
        #                 * np.sin(self.QSX7 * IA + self.QSX8 * np.atan2(self.QSX9 * FY / FZ0, 1)) + self.QSX10
        #                 * np.atan2(self.QSX11 * FZ / FZ0, 1) * IA) * self.LMX
        else:

            # overturning couple (MF-Tyre 6.2 equation manual) -- FZ trig functions do not get corrected to degrees
            MX = (R0 * FZ * self.LMX * (self.QSX1 * self.LVMX - self.QSX2 * IA * (1.0 + self.PPMX1 * dpi)
                                       + self.QSX3 * (FY / FZ0) + self.QSX4 * np.cos(self.QSX5 * np.atan2((self.QSX6 * (FZ / FZ0)) ** 2, 1))
                                       * self.sin(self.QSX7 * IA + self.QSX8 * np.atan2(self.QSX9 * (FY / FZ0), 1))
                                       + self.QSX10 * self.atan(self.QSX11 * (FZ / FZ0)) * IA) + R0 * self.LMX
                  * (FY * (self.QSX13 + self.QSX14 * np.abs(IA)) - FZ * self.QSX12 * IA * np.abs(IA)))

        # correct MX for low FZ values (empirically discovered by Marco Furlan)
        corr = FZ * (FZ / self.FZMIN) ** 2
        MX = self.signals._correct_signal(MX, correction_factor=corr, helper_sig=FZ, threshold=self.FZMIN,
                                            condition="<")

        return MX

    def __my_main_routine(
            self,
            *,
            SL:  SignalLike,
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

        # flip sign for negative speeds -- VCX = VX
        MY = self.signals._flip_negative(MY, helper_sig=VCX)

        # low speed correction (empirically discovered by Marco Furlan, modified by Niek van Rossem)
        VCX_prime = self.correction._find_vc_prime(VC=VCX)
        limit_high = self.VXLOW / VCX_prime - 1.0
        limit_low  = - 1.0 - self.VXLOW - limit_high
        #idx = np.where(SL >= limit_low & SL <= limit_high)
        idx = np.logical_and(self.signals._find_in_signal(SL, condition="<=", threshold=limit_high),
                             self.signals._find_in_signal(SL, condition=">=", threshold=limit_low))
        if isinstance(idx, np.ndarray) or (isinstance(idx, bool) and idx is True):
            x0 = -1.0 * np.ones_like(idx)
            y0 = np.zeros_like(idx)
            x1 = limit_high[idx]
            y1 = np.pi / 2 * np.ones_like(idx)
            speed_correction = self.__interpolate(x0=x0, y0=y0, x1=x1, y1=y1, x=SL)
            MY[idx] *= np.sin(speed_correction)

        # apply correction for slip ratio below the lower limit
        #idx = np.where(SL < limit_low)
        #MY[idx] = - MY[idx]
        MY = self.signals._flip_negative(MY, helper_sig=(limit_low - SL))

        # apply correction for low FZ
        fz_correction = FZ ** 2 / self.FZMIN
        MY = self.signals._correct_signal(MY, correction_factor=fz_correction, helper_sig=FZ, threshold=self.FZMIN,
                                            condition="<")

        return MY

    def _mz_main_routine(
            self,
            *,
            SA:     SignalLike,
            SL:     SignalLike,
            FZ:     SignalLike,
            P:      SignalLike,
            IA:     SignalLike,
            VX:     SignalLike,
            VC:     SignalLike,
            VCX:    SignalLike,
            VS:     SignalLike,
            N:      SignalLike,
            zeta_0: SignalLike,
            zeta_2: SignalLike,
            zeta_4: SignalLike,
            zeta_6: SignalLike,
            zeta_7: SignalLike,
            zeta_8: SignalLike,
            combined_slip: bool = False
    ) -> SignalLike:
        """Function containing the main ``MZ`` calculation routine. Used in ``find_mz`` and ``find_mz_pure``."""

        # unpack tyre properties
        R0 = self.UNLOADED_RADIUS
        V0 = self.LONGVL

        # _normalize pressure and load
        dfz = self.normalize._find_dfz(FZ)
        dpi = self.normalize._find_dpi(P)

        # corrected camber angle
        gamma_star = self.correction._find_gamma_star(IA)

        # friction scaling factors
        LMUY_star  = self.correction._find_lmu_star(VS=VS, V0=V0, LMU=self.LMUY)
        LMUY_prime = self.correction._find_lmu_prime(LMUY_star)

        # cornering stiffness
        KYA  = self.gradient._find_cornering_stiffness(SA=SA, SL=SL, FZ=FZ, N=N, P=P, VX=VX)
        KYA_sign = self.signals._replace_value(np.sign(KYA), target_sig=KYA, target_val=0.0, new_val=1.0)

        # corrected cornering stiffness (4.E39)
        KYA_prime = KYA + self._eps_kappa * KYA_sign

        # camber stiffness
        KYCO = self.gradient._find_camber_stiffness(FZ=FZ, P=P)

        # vertical shift for side force (4.E29)
        S_VY, S_VYg = self.common._find_s_vy(FZ=FZ, VX=VX, dfz=dfz, gamma_star=gamma_star, LMUY_prime=LMUY_prime,
                                             zeta_2=zeta_2)

        # horizontal shift (4.E27)
        S_HY = self.common._find_s_hy(VX=VX, dfz=dfz, KYA=KYA, KYCO=KYCO, gamma_star=gamma_star, S_VYg=S_VYg,
                                      zeta_0=zeta_0, zeta_4=zeta_4)

        # horizontal shift for residual couple (4.E38)
        S_HF = S_HY + S_VY / KYA_prime

        # corrected slip angles (4.E3, 4.E37)
        alpha_star = self.correction._find_alpha_star(SA=SA, VCX=VCX)
        alpha_r = alpha_star + S_HF

        # correction on the slip angle for combined slip
        if combined_slip:

            # slip stiffness
            KXK = self.gradient._find_slip_stiffness(FZ=FZ, P=P)

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
        mu_y = self.friction._find_mu_y(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VX=VX)

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

    @staticmethod
    def __interpolate(
            *,
            x0 : SignalLike,
            y0 : SignalLike,
            x1 : SignalLike,
            y1 : SignalLike,
            x  : SignalLike,
    ) -> SignalLike:
        """
        Interpolates a signal.

        Parameters
        ----------
        *
        x0 : SignalLike
            Lower independent value
        y0 : SignalLike
            Lower dependent value
        x1 : SignalLike
            Higher independent value
        y1 : SignalLike
            Higher dependent value
        x : SignalLike
            Interpolation point

        Returns
        -------
        y : SignalLike
            Interpolated value
        """

        y = (y0 * (x1 - x) * y1 * (x - x0)) / (x1 - x0)
        return y
