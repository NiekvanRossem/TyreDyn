from tyredyn.types.aliases import SignalLike, AngleUnit
from typing import Literal
import numpy as np

class TrailMF6x:
    """
    Pneumatic trail module for the MF 6.1 and MF 6.2 tyre models.
    """

    def __init__(self, model):
        """Import the properties of the overarching ``MF61`` or ``MF62`` class."""
        self._model = model

        # helper functions
        self.correction = model.correction
        self.normalize  = model.normalize
        self.common     = model.common

        # other subsystems
        self.turn_slip  = model.turn_slip

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    def find_trail_pure(
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
        Finds the pneumatic trail of the tyre for pure slip conditions.

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
        t0 : SignalLike
            Pneumatic trail.
        """

        # find other velocity components
        VS, VC = self.extra_signals._find_speeds(SA=SA, SL=0.0, VX=VX)
        VCX = VX

        # turn slip correction
        if self._use_turn_slip:
            PHI = self.extra_signals._find_phi(FZ=FZ, N=N, VC=VC, IA=IA, PHIT=PHIT)
            zeta_5 = self.turn_slip._find_zeta_5(PHI)
        else:
            zeta_5 = self.zeta_default

        # cosine term correction factor
        cos_prime_alpha = self.correction._find_cos_prime_alpha(VC=VC, VCX=VCX)

        # find coefficients
        BT, CT, DT, ET, alpha_t = self.__trail_main_routine(SA=SA, FZ=FZ, P=P, IA=IA, VCX=VCX, VS=VS, zeta_5=zeta_5)

        # pneumatic trail (4.E33)
        t0 = DT * np.cos(CT * self.atan(BT * alpha_t - ET * (BT * alpha_t - self.atan(BT * alpha_t)))) * cos_prime_alpha

        return t0

    def find_trail_combined(
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
        Finds the pneumatic trail of the tyre for combined slip conditions.

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
        t : SignalLike
            Pneumatic trail.
        """

        # find other velocity components
        VS, VC = self.extra_signals._find_speeds(SA=SA, SL=SL, VX=VX)
        VCX = VX

        # turn slip correction
        if self._use_turn_slip:
            PHI = self.extra_signals._find_phi(FZ=FZ, N=N, VC=VC, IA=IA, PHIT=PHIT)
            zeta_5 = self.turn_slip._find_zeta_5(PHI)
        else:
            zeta_5 = self.zeta_default

        # cosine term correction factor
        cos_prime_alpha = self.correction._find_cos_prime_alpha(VC=VC, VCX=VCX)

        # find coefficients
        BT, CT, DT, ET, alpha_t = self.__trail_main_routine(SA=SA, FZ=FZ, P=P, IA=IA, VCX=VCX, VS=VS, zeta_5=zeta_5)

        # cornering stiffness
        KYA = self.gradient._find_cornering_stiffness(SA=SA, SL=SL, FZ=FZ, N=N, P=P, VX=VX)
        KYA_sign = self.signals._replace_value(np.sign(KYA), target_sig=KYA, target_val=0.0, new_val=1.0)

        # slip stiffness
        KXK = self.gradient._find_slip_stiffness(FZ=FZ, P=P)

        # corrected cornering stiffness (4.E39)
        KYA_prime = KYA + self._eps_kappa * KYA_sign

        # corrected slip angle (A55)
        alpha_t_eq = self.atan(np.sqrt(self.tan(alpha_t) ** 2 + (KXK / KYA_prime) ** 2 * SL ** 2)) * np.sign(alpha_t)

        # NOTE: Equation (4.E77) from the book (shown below) does not match the TNO solver, and thus equation (A55) from
        # the paper is used instead (via Marco Furlan).
        # alpha_t_eq = np.sqrt(alpha_t ** 2 + (KXK / KYA_prime) ** 2 * SL ** 2) * np.sign(alpha_t)

        # pneumatic trail (4.E73)
        t = (DT * self.cos(CT * self.atan(BT * alpha_t_eq - ET * (BT * alpha_t_eq - self.atan(BT * alpha_t_eq))))
             * cos_prime_alpha * self.LFZO)

        # NOTE: the trail above is multiplied with LFZO to match the TNO solver. This is not in any official documents,
        # but was discovered by Marco Furlan.

        return t

    #------------------------------------------------------------------------------------------------------------------#
    # INTERNAL FUNCTIONS

    def __trail_main_routine(
            self,
            *,
            SA:     SignalLike,
            FZ:     SignalLike,
            P:      SignalLike,
            IA:     SignalLike,
            VCX:    SignalLike,
            VS:     SignalLike,
            zeta_5: SignalLike,
    ) -> list[SignalLike]:
        """Function containing the main calculations for the pneumatic trail. To be used in ``find_trail`` and
        ``find_trail_pure``."""

        # unpack tyre properties
        R0 = self.UNLOADED_RADIUS
        V0 = self.LONGVL

        # scaled nominal load
        FZ0 = self.FNOMIN
        FZ0_prime = FZ0 * self.LFZO

        # _normalize pressure and load
        dfz = self.normalize._find_dfz(FZ)
        dpi = self.normalize._find_dpi(P)

        # corrected camber angle
        gamma_star = self.correction._find_gamma_star(IA)

        # degressive friction factor (4.E8)
        LMUY_star  = self.correction._find_lmu_star(VS=VS, V0=V0, LMU=self.LMUY)

        # stiffness factor (A58)
        BT = ((self.QBZ1 + self.QBZ2 * dfz + self.QBZ3 * dfz ** 2) * (1.0 + self.QBZ4 * IA + self.QBZ5 * np.abs(IA))
              * self.LYKA / LMUY_star)

        # NOTE: equation (4.E40) in the book (shown in a comment below) does not match the TNO solver. Instead, Equation
        # (A58) from the paper is used above. The parameter QBZ6 is changed to QBZ4 as the former does not exist in TIR
        # files (via Marco Furlan).
        # BT = ((self.QBZ1 + self.QBZ2 * dfz + self.QBZ3 * dfz ** 2) * (1.0 + self.QBZ5 * np.abs(gamma_star) + self.QBZ6 * gamma_star ** 2) * self.LYKA / LMUY_star)

        # shape factor (4.E41)
        CT = self.QCZ1

        # peak factor(A60)
        DT = ((self.QDZ1 + self.QDZ2 * dfz) * (1.0 - self.PPZ1 * dpi) * (1.0 + self.QDZ3 * IA + self.QDZ4 * IA ** 2)
              * FZ * (R0 / FZ0_prime) * self.LTR * zeta_5)

        # NOTE: equation above is taken from the paper instead of the book. Equation (4.E43) from the book (shown in a
        # comment below) does not match the TNO solver (via Marco Furlan):
        # DT = DT0 * (1.0 + self.QDZ3 * np.abs(gamma_star) + self.QDZ4 * gamma_star ** 2) * zeta_5
        # DT0 can be found in the method _find_dt0() in CommonMF6x

        # horizontal shift (4.E35)
        S_HT = self.QHZ1 + self.QHZ2 * dfz + (self.QHZ3 + self.QHZ4 * dfz) * gamma_star

        # corrected slip angle (4.E34)
        alpha_star = self.correction._find_alpha_star(SA=SA, VCX=VCX)
        alpha_t = alpha_star + S_HT

        # curvature factor (4.E44)
        ET = (self.QEZ1 + self.QEZ2 * dfz + self.QEZ3 * dfz ** 2) * (1.0 + (self.QEZ4 + self.QEZ5 * gamma_star)
                                                                     * np.pi / 2 * self.atan(BT * CT * alpha_t))

        return [BT, CT, DT, ET, alpha_t]
