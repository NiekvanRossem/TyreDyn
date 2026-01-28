from tyredyn.types.aliases import SignalLike, AngleUnit
from tyredyn.infrastructure.subsystem_base import SubSystemBase
import numpy as np
from typing import Literal

class GradientsMF6x(SubSystemBase):
    """
    Module containing the gradient functions such as cornering stiffness or camber stiffness for the MF-Tyre 6.1 and MF-Tyre 6.2
    tyre models.
    """

    def _connect(self, model):

        # helper functions
        self.correction = model.correction
        self.normalize  = model.normalize
        self.common     = model.common

        # other subsystems
        self.turn_slip  = model.turn_slip

    def _find_cornering_stiffness(
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
        Returns the gradient of the side force to slip angle, at zero slip.

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
        KYA : SignalLike
            Cornering stiffness at zero slip angle.
        """

        # find other velocity components
        VS, VC = self.extra_signals._find_speeds(SA=SA, SL=SL, VX=VX)

        # unpack tyre properties
        FZ0 = self.FNOMIN

        # turn slip correction
        if self._use_turn_slip:
            PHI = self.extra_signals._find_phi(FZ=FZ, N=N, VC=VC, IA=IA, PHIT=PHIT)
            zeta_3 = self.turn_slip._find_zeta_3(PHI)
        else:
            zeta_3 = self.zeta_default

        # corrected camber angle
        gamma_star = self.correction._find_gamma_star(IA)

        # _normalize pressure
        dpi = self.normalize._find_dpi(P)

        # scaled nominal load
        FZ0_prime = FZ0 * self.LFZO

        # cornering stiffness (4.E25)
        KYA = (self.PKY1 * FZ0_prime * (1.0 + self.PPY1 * dpi) * (1.0 - self.PKY3 * np.abs(gamma_star))
               * self.sin(self.PKY4 * np.atan2(FZ / FZ0_prime, (self.PKY2 + self.PKY5 * gamma_star ** 2)
                                             * (1.0 + self.PPY2 * dpi)))) * zeta_3 * self.LKY
        return KYA

    def _find_slip_stiffness(
            self,
            *,
            FZ: SignalLike,
            P:  SignalLike = None,
    ) -> SignalLike:
        """
        Returns the gradient of the longitudinal force to longitudinal slip, at zero slip ratio.

        Parameters
        ----------
        FZ : SignalLike
            Vertical load.
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).

        Returns
        -------
        KXK : SignalLike
            Slip stiffness at zero slip.
        """

        # _normalize pressure and load
        dfz = self.normalize._find_dfz(FZ)
        dpi = self.normalize._find_dpi(P)

        # slip stiffness (4.E15)
        KXK = (FZ * (self.PKX1 + self.PKX2 * dfz) * np.exp(self.PKX3 * dfz)
               * (1.0 + self.PPX1 * dpi + self.PPX2 * dpi ** 2) * self.LKX)
        return KXK

    def _find_camber_stiffness(
            self,
            *,
            FZ: SignalLike,
            P:  SignalLike = None
    ) -> SignalLike:
        """
        Returns the gradient of the side force to inclination angle.

        Parameters
        ----------
        FZ : SignalLike
            Vertical load.
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).

        Returns
        -------
        KYCO : SignalLike
            Camber stiffness at zero inclination angle.
        """

        # _normalize pressure and load
        dfz = self.normalize._find_dfz(FZ)
        dpi = self.normalize._find_dpi(P)

        # camber stiffness (4.E30)
        KYCO = FZ * (self.PKY6 + self.PKY7 * dfz) * (1.0 - self.PPY5 * dpi) * self.LKYC
        return KYCO

    def _find_aligning_stiffness(
            self,
            *,
            FZ:   SignalLike,
            N:    SignalLike,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
            PHIT: SignalLike = 0.0
    ) -> SignalLike:
        """
        Returns the gradient of self-aligning couple to slip angle, at zero slip.

        Parameters
        ----------
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
        KZAO : SignalLike
            Self-aligning couple stiffness to slip angle.
        """

        # find other velocity components
        VCX = VX

        # unpack tyre properties
        R0 = self.UNLOADED_RADIUS

        # corrected nominal load
        FZ0 = self.FNOMIN
        FZ0_prime = FZ0 * self.LFZO

        # _normalize pressure and load
        dfz = self.normalize._find_dfz(FZ)
        dpi = self.normalize._find_dpi(P)

        # cornering stiffness
        KYA = self._find_cornering_stiffness(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

        # static trail peak factor
        D_T0 = self.common._find_dt0(FZ=FZ, dfz=dfz, dpi=dpi, VCX=VCX, FZ0_prime=FZ0_prime, R0=R0)

        # cornering stiffness to self aligning couple (4.E48)
        # NOTE: not used generally, but added for completeness
        KZAO = D_T0 * KYA
        return KZAO

    def _find_camber_aligning_stiffness(
            self,
            *,
            FZ:  SignalLike,
            P:   SignalLike = None,
            VX:  SignalLike = None
    ) -> SignalLike:
        """
        Returns the gradient of the self-aligning couple to camber angle, at zero camber.

        Parameters
        ----------
        FZ : SignalLike
            Vertical load.
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        VX : SignalLike, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).

        Returns
        -------
        KZCO : SignalLike
            Self-aligning couple stiffness to camber angle.
        """

        # find other velocity components
        VS, VC = self.extra_signals._find_speeds(SA=SA, SL=SL, VX=VX)
        VCX = VX

        # unpack tyre properties
        R0 = self.UNLOADED_RADIUS
        V0 = self.LONGVL

        # _normalize pressure and load
        dfz = self.normalize._find_dfz(FZ)
        dpi = self.normalize._find_dpi(P)

        # degressive friction factor with speed
        LMUY_star = self.correction._find_lmu_star(VS=VS, V0=V0, LMU=self.LMUY)

        # static trail peak factor
        D_T0 = self.common._find_dt0(FZ=FZ, dfz=dfz, dpi=dpi, VCX=VCX, FZ0_prime=FZ0_prime, R0=R0)

        # camber stiffness to slip angle at zero slip
        KYCO = self._find_camber_stiffness(FZ=FZ, P=P)

        # camber stiffness to self aligning couple (4.E49)
        # NOTE: not used generally, but added for completeness
        KZCO = FZ * R0 * (self.QDZ8 + self.QDZ9 * dfz) * (1.0 + self.PPZ2 * dpi) * self.LKZC * LMUY_star - D_T0 * KYCO
        return KZCO

    # TODO: find better method for the functions below
    @staticmethod
    def find_instant_kya(
            *,
            SA: SignalLike,
            FY: SignalLike
    ) -> SignalLike:
        """
        Returns the instantaneous cornering stiffness of the tyre by calculating the gradient of the lateral force and
        the slip angle.

        Parameters
        ----------
        SA : SignalLike
            Slip angle.
        FY : SignalLike
            Side force.

        Returns
        -------
        iKYA : SignalLike
            Instantaneous cornering stiffness
        """

        # instantaneous cornering stiffness (not defined in the book, method from MFeval)
        iKYA = np.gradient(FY, SA)
        return iKYA

    @staticmethod
    def find_instant_kxk(
            *,
            SL: SignalLike,
            FX: SignalLike
    ) -> SignalLike:
        """
        Returns the instantaneous slip stiffness of the tyre by calculating the gradient of the longitudinal force and
        the slip ratio.

        Parameters
        ----------
        SL : SignalLike
            Slip ratio.
        FX : SignalLike
            Longitudinal force.

        Returns
        -------
        iKXK : SignalLike
            Instantaneous slip stiffness
        """

        # instantaneous slip stiffness from MFeval)
        iKXK = np.gradient(FX, SL)
        return iKXK

