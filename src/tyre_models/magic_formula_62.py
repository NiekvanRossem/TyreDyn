from src.utils.formatting import SignalLike, AngleUnit
from src.tyre_models.base_tyre import TyreBase

# import helper modules
from src.helpers.common_mf61 import CommonMF61
from src.helpers.corrections import CorrectionsMF61
from src.helpers.normalize import Normalize

# import modules
from src.modules.contact_patch.contact_patch_mf61 import ContactPatchMF61
from src.modules.forces.forces_mf61 import ForcesMF61
from src.modules.friction_coefficient.friction_mf61 import FrictionMF61
from src.modules.gradients.gradients_mf61 import GradientsMF61
from src.modules.moments.moments_mf61 import MomentsMF61
from src.modules.radius_and_deflection.radius_mf62 import RadiusMF62
from src.modules.relaxation.relaxation_mf61 import RelaxationMF61
from src.modules.trail.trail_mf61 import TrailMF61
from src.modules.turn_slip.turn_slip import TurnSlip
from src.modules.stiffness.stiffness_mf61 import StiffnessMF61

from functools import wraps
from typing import Literal

class MF62(TyreBase):
    """
    Class definition for the Magic Formula 6.2 tyre model. Initialize an instance of this class by calling
    ``Tyre(<filename>.tir)``, where ``<filename>.tir`` is a TIR property file with ``FITTYP`` ``62`` or newer.

    This class contains functions to evaluate the tyre state based on a set of inputs. Equations are mainly based on the
    MF 6.2 equation manual by TNO. Some equations are taken from Besselink's 2010 paper in order to match the TNO
    solver and MFeval. Corrections from Marco Furlan.

    References:
      - Pacejka, H.B. & Besselink, I.J.M. (2012). *Tire and Vehicle Dynamics. Third Edition*. Elsevier.
        `doi: 10.1016/c2010-0-68548-8 <https://doi.org/10.1016/c2010-0-68548-8>`_
      - Besselink, I.J.M. & Schmeitz, A.J.C. & Pacejka, H.B. (2010). *An improved Magic Formula/Swift tyre model that
        can handle inflation pressure changes*. Vehicle System Dynamics, 48(sup1), 337–352.
        `doi: 10.1080/00423111003748088 <https://doi-org.tudelft.idm.oclc.org/10.1080/00423111003748088>`_
      - Besselink, I.J.M. & Schmeitz, A.J.C. & Pacejka, H.B. (2010). *An improved Magic Formula/Swift tyre model that
        can handle inflation pressure changes* [Unpublished manuscript]. Retrieved 30 December 2025.
        `https://pure.tue.nl/ws/files/3139488/677330157969510.pdf <https://pure.tue.nl/ws/files/3139488/677330157969510.pdf>`_
      - Marco Furlan (2025). *MFeval*. MATLAB Central File Exchange. Retrieved December 18, 2025.
        `mathworks.com/matlabcentral/fileexchange/63618-mfeval <https://mathworks.com/matlabcentral/fileexchange/63618-mfeval>`_
      - Netherlands Organization for Applied Scientific Research (2013). *MF-Tyre/MF-Swift 6.2 -- Equation manual.
        Document revision 20130706.*
      - International Organization for Standardization (2011). *Road vehicles -- Vehicle dynamics and road-holding
        ability -- Vocabulary* (ISO standard No. 8855:2011)
        `iso.org/obp/ui/#iso:std:iso:8855:ed-2:v1:en <https://www.iso.org/obp/ui/#iso:std:iso:8855:ed-2:v1:en>`_
      - LeMesurier, B. & Roberts, S. (2025). *Numerical Methods and Analysis with Python -- Root finding without
        derivatives*. College of Charleston. Retrieved 28 December 2025.
        `lemesurierb.people.charleston.edu/numerical-methods-and-analysis-python/main/root-finding-without-
        derivatives-python.html <https://lemesurierb.people.charleston.edu/numerical-methods-and-analysis-python/
        main/root-finding-without-derivatives-python.html>`_
    """

    def __init__(self, data, **settings):

        # run the initialization from the tyre_models class
        super().__init_from_data__(data, **settings)

        # default value if no turn slip is selected
        self.zeta_default = 1.0

        # correction factors to avoid singularities at low speed
        self.eps_r      = 1e-6
        self.eps_x      = 1e-6
        self.eps_kappa  = 1e-6
        self.eps_V      = 0.1  # set to 0.1 as suggested by Pacejka

        # scaling factor to control decaying friction with increasing speed (set to zero generally)
        self.LMUV = 0.0

        # low friction correction for friction coefficient scaling factor, set to 10 as suggested by Pacejka (4.E8)
        self.A_mu = 10.0

        # import helper functions
        self.normalize      = Normalize(self)
        self.correction     = CorrectionsMF61(self)     # depends on normalize
        self.common         = CommonMF61(self)          # depends on normalize and correction

        # import modules (order is important here since some modules depend on others)
        self.friction       = FrictionMF61(self)        # depends only on helper functions
        self.stiffness      = StiffnessMF61(self)       # depends only on helper functions
        self.turn_slip      = TurnSlip(self)            # depends on friction and gradients
        self.contact_patch  = ContactPatchMF61(self)    # depends on stiffness
        self.radius         = RadiusMF62(self)          # depends on stiffness
        self.trail          = TrailMF61(self)           # depends on turn slip
        self.gradient       = GradientsMF61(self)       # depends on turn slip
        self.relaxation     = RelaxationMF61(self)      # depends on stiffness and gradient
        self.forces         = ForcesMF61(self)          # depends on turn slip, gradient, and forces
        self.moments        = MomentsMF61(self)         # depends on turn slip, friction, trail, gradient, and forces

    #------------------------------------------------------------------------------------------------------------------#
    # STATE OUTPUTS

    def find_forces(
            self,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            *,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VC:   SignalLike = None,
            VCX:  SignalLike = None,
            VS:   SignalLike = 0.0,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> list[SignalLike]:
        """
        Finds the force vector for combined slip conditions. Order is ``FX``, ``FY``, ``FZ``.

        Parameters
        ----------
        VC
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
            Contact patch slip speed (will default to zero if not specified).
        PHIT : SignalLike, optional
            Turn slip (will default to zero if not specified).
        angle_unit : str, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        FX, FY, FZ : list[SignalLike]
            Force vector.
        """

        # find planar forces
        FX = self.forces.find_fx_combined(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VC=VC, VCX=VCX, VS=VS, angle_unit=angle_unit)
        FY = self.forces.find_fy_combined(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VCX=VCX, VS=VS, PHIT=PHIT, angle_unit=angle_unit)
        return [FX, FY, FZ]

    def find_moments(
            self,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            *,
            VX:   SignalLike = None,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VC:   SignalLike = None,
            VCX:  SignalLike = None,
            VS:   SignalLike = 0.0,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> list[SignalLike]:
        """
        Finds the moment vector for combined slip conditions. Order is ``MX``, ``MY``, ``MZ``.

        Parameters
        ----------
        SA : SignalLike
            Slip angle.
        SL : SignalLike
            Slip ratio.
        FZ : SignalLike
            Vertical load.
        VX : SignalLike, optional
            Wheel centre longitudinal speed (will default to ``VCX`` if not specified, if ``VCX`` is also not specified
            it will default to ``LONGVL``).
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : SignalLike, optional
            Inclination angle with respect to the ground plane (will default to zero if not specified).
        VC : SignalLike, optional
            Contact patch speed (will default to ``LONGVL`` if not specified).
        VCX : SignalLike, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).
        VS : SignalLike, optional
            Contact patch slip speed (will default to zero if not specified).
        PHIT : SignalLike, optional
            Turn slip (will default to zero if not specified).
        angle_unit : str, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        MX, MY, MZ : list[SignalLike]
            Moment vector.
        """

        VX = VCX if VCX is not None else VX

        MX = self.moments.find_mx_combined(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VCX=VCX, VS=VS, PHIT=PHIT, angle_unit=angle_unit)
        MY = self.moments.find_my_combined(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VC=VC, VCX=VX, VS=VS, PHIT=PHIT,
                                           angle_unit=angle_unit)
        MZ = self.moments.find_mz_combined(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VC=VC, VCX=VCX, VS=VS, PHIT=PHIT, angle_unit=angle_unit)
        return [MX, MY, MZ]

    def find_force_moment(
            self,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            VX:   SignalLike,
            *,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VC:   SignalLike = None,
            VCX:  SignalLike = None,
            VS:   SignalLike = 0.0,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> list[SignalLike]:
        """
        Finds the total force and moment vector of the tyre for combined slip conditions. Order is ``FX``, ``FY``,
        ``FZ``, ``MX``, ``MY``, ``MZ``.

        Parameters
        ----------
        SA : SignalLike
            Slip angle.
        SL : SignalLike
            Slip ratio.
        FZ : SignalLike
            Vertical load.
        VX : SignalLike, optional
            Wheel centre longitudinal speed (will default to ``VCX`` if not specified, if ``VCX`` is also not specified
            it will default to ``LONGVL``).
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : SignalLike, optional
            Inclination angle with respect to the ground plane (will default to zero if not specified).
        VC : SignalLike, optional
            Contact patch speed (will default to ``LONGVL`` if not specified).
        VCX : SignalLike, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).
        VS : SignalLike, optional
            Contact patch slip speed (will default to zero if not specified).
        PHIT : SignalLike, optional
            Turn slip (will default to zero if not specified).
        angle_unit : str, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        FX, FY, FZ, MX, MY, MZ : list[SignalLike]
            Force and moment vector.
        """

        [FX, FY, FZ] = self.find_forces(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VC=VC, VCX=VCX, VS=VS, PHIT=PHIT,
                                        angle_unit=angle_unit)
        [MX, MY, MZ] = self.find_moments(SA=SA, SL=SL, FZ=FZ, VX=VX, P=P, IA=IA, VC=VC, VCX=VCX, VS=VS, PHIT=PHIT, angle_unit=angle_unit)
        return [FX, FY, FZ, MX, MY, MZ]

    def find_lateral_output(
            self,
            SA:   SignalLike,
            FZ:   SignalLike,
            N:    SignalLike,
            *,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VC:   SignalLike = None,
            VCX:  SignalLike = None,
            VS:   SignalLike = 0.0,
            PHIT: SignalLike = 0.0,
            angle_unit: Literal["rad", "deg"] = "rad"
    ) -> list[SignalLike]:
        """
        Finds the free rolling outputs commonly used in lateral vehicle tyre_models. Order is ``FY``, ``MX``, ``MZ``,
        ``RL``, ``sigma_y``.

        Parameters
        ----------
        SA : SignalLike
            Slip angle.
        FZ : SignalLike
            Vertical load.
        N : SignalLike
            Angular speed of the wheel.
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : SignalLike, optional
            Inclination angle with respect to the ground plane (will default to zero if not specified).
        VC : SignalLike, optional
            Contact patch speed (will default to ``LONGVL`` if not specified).
        VCX : SignalLike, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).
        VS : SignalLike, optional
            Contact patch slip speed (will default to zero if not specified).
        PHIT : SignalLike, optional
            Turn slip (will default to zero if not specified).
        angle_unit : str, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        FY, MX, MZ, RL, sigma_y: list[SignalLike]
            Lateral output vector.
        """

        FY = self.forces.find_fy_pure(SA=SA, FZ=FZ, P=P, IA=IA, VCX=VCX, VS=VS, PHIT=PHIT, angle_unit=angle_unit)
        MX = self.moments.find_mx_pure(SA=SA, FZ=FZ, P=P, IA=IA, VCX=VCX, VS=VS, PHIT=PHIT, angle_unit=angle_unit)
        MZ = self.moments.find_mz_pure(SA=SA, FZ=FZ, P=P, IA=IA, VC=VC, VCX=VCX, VS=VS, PHIT=PHIT, angle_unit=angle_unit)
        _, _, RL, _ = self.radius.find_radius(FX=0.0, FY=FY, FZ=FZ, N=N, P=P, IA=IA)
        sigma_y = self.find_lateral_relaxation(FZ=FZ, P=P, IA=IA, PHIT=PHIT, angle_unit=angle_unit)

        return [FY, MX, MZ, RL, sigma_y]

    def find_longitudinal_output(
            self,
            SL:   SignalLike,
            FZ:   SignalLike,
            N:    SignalLike,
            *,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VC:   SignalLike = None,
            VS:   SignalLike = 0.0,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> list[SignalLike]:
        """
        Finds the output signals commonly used in longitudinal vehicle tyre_models. Order is ``FX``, ``MY``, ``RL``,
        ``RE``, ``sigma_x``.

        Parameters
        ----------
        VC
        SL : SignalLike
            Slip ratio.
        FZ : SignalLike
            Vertical load.
        N : SignalLike
            Angular speed of the wheel.
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : SignalLike, optional
            Inclination angle with respect to the ground plane (will default to zero if not specified).
        VS : SignalLike, optional
            Contact patch slip speed (will default to zero if not specified).
        PHIT : SignalLike, optional
            Turn slip (will default to zero if not specified).
        angle_unit : str, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        FX, MY, RL, RE, sigma_x: list[SignalLike]
            Longitudinal output vector.
        """

        FX = self.forces.find_fx_pure(SL=SL, FZ=FZ, P=P, IA=IA, VC=VC, VS=VS, PHIT=PHIT, angle_unit=angle_unit)
        MY = self.forces.find_my_pure(SL=SL, FZ=FZ, P=P, IA=IA, VS=VS, angle_unit=angle_unit)
        _, RE, RL, _ = self.radius.find_radius(FX=FX, FY=FY, FZ=FZ, N=N, P=P, IA=IA)
        sigma_x = self.relaxation.find_longitudinal_relaxation(FZ=FZ, P=P)
        return [FX, MY, RL, RE, sigma_x]

    def find_full_output(
            self,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            VX:   SignalLike,
            N:    SignalLike,
            *,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VC:   SignalLike = None,
            VCX:  SignalLike = None,
            VS:   SignalLike = 0.0,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> list[SignalLike]:
        """
        Finds the full output state of the tyre. Not recommended to use this in performance-sensitive vehicle
        simulation, as some functions are called multiple times.

        Parameters
        ----------
        SA : SignalLike
            Slip angle.
        SL : SignalLike
            Slip ratio.
        FZ : SignalLike
            Vertical load.
        VX : SignalLike
            Wheel centre longitudinal speed.
        N : SignalLike
            Angular speed of the wheel.
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : SignalLike, optional
            Inclination angle with respect to the ground plane (will default to zero if not specified).
        VC : SignalLike, optional
            Contact patch speed (will default to ``VX`` if not specified).
        VCX : SignalLike, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).
        VS : SignalLike, optional
            Contact patch slip speed (will default to zero if not specified).
        PHIT : SignalLike, optional
            Turn slip (will default to zero if not specified).
        angle_unit : str, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        OUT: list[SignalLike]
            Full output state.
        """

        # set default values for optional arguments
        P   = self.INFLPRES if P is None else P
        VC  = self.LONGVL if VC is None else VC
        VCX = self.LONGVL if VCX is None else VCX

        # turn slip correction
        if self._use_turn_slip:
            PHI = self.correction._find_phi(FZ=FZ, N=N, VC=VC, IA=IA, PHIT=PHIT)
            zeta_0 = 0.0
            zeta_2 = self.turn_slip._find_zeta_2(SA=SA, FZ=FZ, PHI=PHI)
            zeta_4 = self.turn_slip._find_zeta_4(FZ=FZ, P=P, IA=IA, VCX=VCX, VS=VS, PHI=PHI, zeta_2=zeta_2, angle_unit=angle_unit)
            zeta_6 = self.turn_slip._find_zeta_6(PHI)
            zeta_7 = self.turn_slip._find_zeta_7(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VCX=VCX, VS=VS, PHI=PHI, angle_unit=angle_unit)
            zeta_8 = self.turn_slip._find_zeta_8(FZ=FZ, P=P, IA=IA, VS=VS, PHI=PHI, angle_unit=angle_unit)
        else:
            zeta_0 = self.zeta_default
            zeta_2 = self.zeta_default
            zeta_4 = self.zeta_default
            zeta_6 = self.zeta_default
            zeta_7 = self.zeta_default
            zeta_8 = self.zeta_default

        # force and moment vector
        [FX, FY, FZ, MX, MY, MZ] = self.find_force_moment(SA=SA, SL=SL, FZ=FZ, VX=VX, P=P, IA=IA, VC=VC, VCX=VCX, VS=VS,
                                                          PHIT=PHIT, angle_unit=angle_unit)

        # residual self-aligning couple
        MZR = self.moments._mz_main_routine(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VC=VC, VCX=VCX, VS=VS,
                                            zeta_0=zeta_0, zeta_2=zeta_2, zeta_4=zeta_4, zeta_6=zeta_6, zeta_7=zeta_7,
                                            zeta_8=zeta_8, combined_slip=True, angle_unit=angle_unit)

        # free, loaded, and effective radii, and deflection
        R_omega, RE, RL, rho = self.radius.find_radius(FX=FX, FY=FY, FZ=FZ, N=N, P=P, IA=IA)

        # pneumatic trail
        t = self.trail.find_trail_combined(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VC=VC, VCX=VCX, VS=VS, PHIT=PHIT, angle_unit=angle_unit)

        # friction coefficients
        mu_x = self.friction.find_mu_x(FZ=FZ, P=P, IA=IA, VS=VS, angle_unit=angle_unit)
        mu_y = self.friction.find_mu_y(FZ=FZ, P=P, IA=IA, VS=VS, angle_unit=angle_unit)

        # contact patch dimensions
        a, b = self.contact_patch.find_contact_patch(FZ=FZ, P=P)

        # tyre stiffness
        Cx = self.stiffness.find_longitudinal_stiffness(FZ=FZ, P=P)
        Cy = self.stiffness.find_lateral_stiffness(FZ=FZ, P=P)
        Cz = self.stiffness.find_vertical_stiffness(P=P)

        # slip stiffness
        KXK = self.gradient.find_slip_stiffness(FZ=FZ, P=P)
        KYA = self.gradient.find_cornering_stiffness(FZ=FZ, P=P, IA=IA, PHIT=PHIT, angle_unit=angle_unit)

        # relaxation length
        sigma_x = self.relaxation.find_longitudinal_relaxation(FZ=FZ, P=P)
        sigma_y = self.relaxation.find_lateral_relaxation(FZ=FZ, P=P, IA=IA, PHIT=PHIT, angle_unit=angle_unit)

        # instantaneous slip stiffness
        iKYA = self.gradient.find_instant_kya(SA=SA, FY=FY)
        iKXK = self.gradient.find_instant_kxk(SL=SL, FX=FX) if self._use_mfeval_mode else None

        # assemble final output
        if self._use_mfeval_mode:

            # compatibility mode. Output vector has the same order as MFeval
            output = [FX, FY, FZ, MX, MY, MZ, SL, SA, IA, PHIT, VX, P, RE, rho, 2 * a,
                      t, mu_x, mu_y, N, RL, 2 * b, MZR, Cx, Cy, Cz, KYA, sigma_x, sigma_y, iKYA, KXK]
        else:

            # more organized output vector
            output = [
                FX, FY, FZ,                 # FORCES
                MX, MY, MZ,                 # MOMENTS
                SL, SA, IA, PHIT, VX, P, N,  # INPUT STATE
                R_omega, RE, rho, RL,       # RADII
                2*a, 2*b,                   # CONTACT PATCH
                t,                          # TRAIL
                mu_x, mu_y,                 # FRICTION COEFFICIENT
                MZR,                        # RESIDUAL MOMENT
                Cx, Cy, Cz,                 # TYRE STIFFNESS
                KYA, iKYA, KXK, iKXK,       # SLIP STIFFNESS
                sigma_x, sigma_y            # RELAXATION LENGTHS
            ]
        return output

    #------------------------------------------------------------------------------------------------------------------#
    # FORCES

    @wraps(ForcesMF61.find_fx_pure)
    def find_fx_pure(
            self,
            SL:   SignalLike,
            FZ:   SignalLike,
            *,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VC:   SignalLike = None,
            VS:   SignalLike = 0.0,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:
        return self.forces.find_fx_pure(SL=SL, FZ=FZ, P=P, IA=IA, VC=VC, VS=VS, PHIT=PHIT, angle_unit=angle_unit)

    @wraps(ForcesMF61.find_fy_pure)
    def find_fy_pure(
            self,
            SA:   SignalLike,
            FZ:   SignalLike,
            *,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VCX:  SignalLike = None,
            VS:   SignalLike = 0.0,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:
        return self.forces.find_fy_pure(SA=SA, FZ=FZ, P=P, IA=IA, VCX=VCX, VS=VS, PHIT=PHIT, angle_unit=angle_unit)

    @wraps(ForcesMF61.find_fx_combined)
    def find_fx_combined(
            self,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            *,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VC:   SignalLike = None,
            VCX:  SignalLike = None,
            VS:   SignalLike = 0.0,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:
        return self.forces.find_fx_combined(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VC=VC, VCX=VCX, VS=VS, PHIT=PHIT,
                                            angle_unit=angle_unit)

    @wraps(ForcesMF61.find_fy_combined)
    def find_fy_combined(
            self,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            *,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VCX:  SignalLike = None,
            VS:   SignalLike = 0.0,
            PHIT: SignalLike = 0.0,
            angle_unit: Literal["deg", "rad"] = "rad"
    ) -> SignalLike:
        return self.forces.find_fy_combined(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VCX=VCX, VS=VS, PHIT=PHIT, angle_unit=angle_unit)

    #------------------------------------------------------------------------------------------------------------------#
    # MOMENTS

    @wraps(MomentsMF61.find_mx_pure)
    def find_mx_pure(
            self,
            SA:   SignalLike,
            FZ:   SignalLike,
            *,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VCX:  SignalLike = None,
            VS:   SignalLike = 0.0,
            PHIT: SignalLike = 0.0,
            angle_unit: Literal["deg", "rad"] = "rad"
    ) -> SignalLike:
        return self.moments.find_mx_pure(SA=SA, FZ=FZ, P=P, IA=IA, VCX=VCX, VS=VS, PHIT=PHIT, angle_unit=angle_unit)

    @wraps(MomentsMF61.find_my_pure)
    def find_my_pure(
            self,
            SL: SignalLike,
            FZ: SignalLike,
            *,
            P:  SignalLike = None,
            IA: SignalLike = 0.0,
            VC: SignalLike = None,
            VX: SignalLike = None,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:
        return self.moments.find_my_pure(SL=SL, FZ=FZ, P=P, IA=IA, VC=VC, VCX=VX, angle_unit=angle_unit)

    @wraps(MomentsMF61.find_mz_pure)
    def find_mz_pure(
            self,
            SA:   SignalLike,
            FZ:   SignalLike,
            *,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VC:   SignalLike = None,
            VCX:  SignalLike = None,
            VS:   SignalLike = 0.0,
            PHIT: SignalLike = 0.0,
         angle_unit: AngleUnit = "rad"
    ) -> SignalLike:
        return self.moments.find_mz_pure(SA=SA, FZ=FZ, P=P, IA=IA, VC=VC, VCX=VCX, VS=VS, PHIT=PHIT, angle_unit=angle_unit)

    @wraps(MomentsMF61.find_mx_combined)
    def find_mx_combined(
            self,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            *,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VCX:  SignalLike = None,
            VS:   SignalLike = 0.0,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:
        return self.moments.find_mx_combined(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VCX=VCX, VS=VS, PHIT=PHIT, angle_unit=angle_unit)

    @wraps(MomentsMF61.find_my_combined)
    def find_my_combined(
            self,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            *,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VC:   SignalLike = None,
            VX:   SignalLike = None,
            VS:   SignalLike = 0.0,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:
        return self.moments.find_my_combined(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VC=VC, VCX=VX, VS=VS, PHIT=PHIT,
                                             angle_unit=angle_unit)

    @wraps(MomentsMF61.find_mz_combined)
    def find_mz_combined(
            self,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            *,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VC:   SignalLike = None,
            VCX:  SignalLike = None,
            VS:   SignalLike = 0.0,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:
        return self.moments.find_mz_combined(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VC=VC, VCX=VCX, VS=VS, PHIT=PHIT, angle_unit=angle_unit)

    #------------------------------------------------------------------------------------------------------------------#
    # RADIUS AND DEFLECTION

    @wraps(RadiusMF62.find_radius)
    def find_radius(
            self,
            FX: SignalLike,
            FY: SignalLike,
            FZ: SignalLike,
            N:  SignalLike,
            *,
            P: SignalLike  = None,
            IA: SignalLike = 0.0,
            maxiter: int   = 30,
            tolx: float    = 1e-6
    ) -> list[SignalLike]:
        return self.radius.find_radius(FX=FX, FY=FY, FZ=FZ, N=N, P=P, IA=IA, maxiter=maxiter, tolx=tolx)

    #------------------------------------------------------------------------------------------------------------------#
    # FRICTION COEFFICIENT

    @wraps(FrictionMF61.find_mu_x)
    def find_mu_x(
            self,
            FZ: SignalLike,
            *,
            P:  SignalLike = None,
            IA: SignalLike = 0.0,
            VS: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:
        return self.friction.find_mu_x(FZ=FZ, P=P, IA=IA, VS=VS, angle_unit=angle_unit)

    @wraps(FrictionMF61.find_mu_y)
    def find_mu_y(
            self,
            FZ: SignalLike,
            *,
            P:  SignalLike = None,
            IA: SignalLike = 0.0,
            VS: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:
        return self.friction.find_mu_y(FZ=FZ, P=P, IA=IA, VS=VS, angle_unit=angle_unit)

    #------------------------------------------------------------------------------------------------------------------#
    # TYRE STIFFNESS

    @wraps(StiffnessMF61.find_lateral_stiffness)
    def find_lateral_stiffness(
            self,
            FZ: SignalLike,
            *,
            P:  SignalLike = None
    ) -> SignalLike:
        return self.stiffness.find_lateral_stiffness(FZ=FZ, P=P)

    @wraps(StiffnessMF61.find_longitudinal_stiffness)
    def find_longitudinal_stiffness(
            self,
            FZ: SignalLike,
            *,
            P: SignalLike = None
    ) -> SignalLike:
        return self.stiffness.find_longitudinal_stiffness(FZ=FZ, P=P)

    @wraps(StiffnessMF61.find_vertical_stiffness)
    def find_vertical_stiffness(
            self,
            P: SignalLike
    ) -> SignalLike:
        return self.stiffness.find_vertical_stiffness(P=P)

    #------------------------------------------------------------------------------------------------------------------#
    # CONTACT PATCH DIMENSIONS

    @wraps(ContactPatchMF61.find_contact_patch)
    def find_contact_patch(
            self,
            FZ: SignalLike,
            *,
            P:  SignalLike = None
    ) -> list[SignalLike]:
        return self.contact_patch.find_contact_patch(FZ=FZ, P=P)

    # ------------------------------------------------------------------------------------------------------------------#
    # PNEUMATIC TRAIL

    @wraps(TrailMF61.find_trail_pure)
    def find_trail_pure(
            self,
            SA:   SignalLike,
            FZ:   SignalLike,
            *,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VC:   SignalLike = None,
            VCX:  SignalLike = None,
            VS:   SignalLike = 0.0,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:
        return self.trail.find_trail_pure(SA=SA, FZ=FZ, P=P, IA=IA, VC=VC, VCX=VCX, VS=VS, PHIT=PHIT, angle_unit=angle_unit)

    @wraps(TrailMF61.find_trail_combined)
    def find_trail_combined(
            self,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            *,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VC:   SignalLike = None,
            VCX:  SignalLike = None,
            VS:   SignalLike = 0.0,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:
        return self.trail.find_trail_combined(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VC=VC, VCX=VCX, VS=VS, PHIT=PHIT, angle_unit=angle_unit)

    #------------------------------------------------------------------------------------------------------------------#
    # GRADIENTS

    @wraps(GradientsMF61.find_cornering_stiffness)
    def find_cornering_stiffness(
            self,
            FZ:   SignalLike,
            *,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:
        return self.gradient.find_cornering_stiffness(FZ=FZ, P=P, IA=IA, PHIT=PHIT, angle_unit=angle_unit)

    @wraps(GradientsMF61.find_slip_stiffness)
    def find_slip_stiffness(
            self,
            FZ: SignalLike,
            *,
            P:  SignalLike = None
    ) -> SignalLike:
        return self.gradient.find_slip_stiffness(FZ=FZ, P=P)

    @wraps(GradientsMF61.find_camber_stiffness)
    def find_camber_stiffness(
            self,
            FZ: SignalLike,
            *,
            P:  SignalLike = None
    ) -> SignalLike:
        return self.gradient.find_camber_stiffness(FZ=FZ, P=P)

    @wraps(GradientsMF61.find_instant_kya)
    def find_cornering_stiffness_instant(
            self,
            SA: SignalLike,
            FY: SignalLike,
    ) -> SignalLike:
        return self.gradient.find_instant_kya(SA=SA, FY=FY)

    @wraps(GradientsMF61.find_instant_kxk)
    def find_slip_stiffness_instant(
            self,
            SL: SignalLike,
            FX: SignalLike
    ) -> SignalLike:
        return self.gradient.find_instant_kxk(SL=SL, FX=FX)

    #------------------------------------------------------------------------------------------------------------------#
    # RELAXATION LENGTHS

    @wraps(RelaxationMF61.find_lateral_relaxation)
    def find_lateral_relaxation(
            self,
            FZ:   SignalLike,
            *,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:
        return self.relaxation.find_lateral_relaxation(FZ=FZ, P=P, IA=IA, PHIT=PHIT, angle_unit=angle_unit)

    @wraps(RelaxationMF61.find_longitudinal_relaxation)
    def find_longitudinal_relaxation(
            self,
            FZ: SignalLike,
            *,
            P:  SignalLike = None
    ) -> SignalLike:
        return self.relaxation.find_longitudinal_relaxation(FZ=FZ, P=P)

# test script
if __name__ == "__main__":

    from src.initialize_tyre import Tyre
    import numpy as np

    # initialize tyre
    tyre = Tyre(
        'car205_60R19.tir',
        validate        = True,
        use_alpha_star  = True,
        use_gamma_star  = True,
        use_model_type  = "MF62",
        use_lmu_star    = True,
        use_turn_slip   = False,
        check_format    = True,
        check_limits    = True,
        use_mfeval_mode = False
    )

    # input state
    SA   = 1.0
    SL   = 0.0
    FZ   = 4500
    P    = 1.8e5
    IA   = 0.0
    VX   = 200 / 3.6
    PHIT = 0.0
    VS   = -SL * VX
    N    = VX / tyre.UNLOADED_RADIUS

    [FX, FY, FZ,
     MX, MY, MZ,
     SL, SA, IA, PHIT, VX, P, N,
     R_omega, RE, rho, RL,
     a, b, t,
     mu_x, mu_y,
     MZR,
     Cx, Cy, Cz,
     KYA, iKYA, KXK, iKXK,
     sigma_x, sigma_y] = tyre.find_full_output(SA, SL, FZ, VX, N, P=P, IA=IA, VC=None, VCX=None, VS=VS, PHIT=PHIT, angle_unit="deg")

    N_new = VX / RE

    def rads2rpm(input):
        return input * 60.0 / (2.0 * np.pi)

    print("=== FULL STATE OUTPUT ===")
    print("Input state")
    print(f"  Slip angle:           {SA:.1f} deg")
    print(f"  Slip ratio:           {SL:.1f}")
    print(f"  Inclination angle:    {IA:.1f} deg")
    print(f"  Tyre pressure:        {1e-5 * P:.2f} bar")
    print(f"  Turn slip:            {PHIT:.1f} /m")

    print("Speed")
    print(f"  Longitudinal:         {3.6 * VX:.1f} km/h")
    print(f"  Angular (old):        {rads2rpm(N):.1f} rpm")
    print(f"  Angular (new):        {rads2rpm(N_new):.1f} rpm")

    print("Forces")
    print(f"  Longitudinal:         {FX:.1f} N")
    print(f"  Lateral:              {-FY:.1f} N")
    print(f"  Vertical:             {FZ:.1f} N")

    print("Moments")
    print(f"  Overturning:          {MX:.1f} Nm")
    print(f"  Rolling resistance:   {MY:.1f} Nm")
    print(f"  Self-aligning:        {MZ:.1f} Nm")
    print(f"  Residual MZ:          {MZR:.1f} Nm")

    print("Gradients")
    print(f"  Cornering stiffness:  {-np.deg2rad(KYA):.1f} N/deg")
    print(f"  Slip stiffness:       {1e-2 * KXK:.1f} N/0.01slip")

    print(f"Friction coefficients")
    print(f"  Longitudinal:         {mu_x:.3f}")
    print(f"  Lateral:              {mu_y:.3f}")

    print("Relaxation lengths")
    print(f"  Longitudinal:         {1e3 * sigma_x:.1f} mm")
    print(f"  Lateral:              {-1e3 * sigma_y:.1f} mm")

    print("Radii and deflection")
    print(f"  Free radius:          {1e3 * R_omega:.1f} mm")
    print(f"  Loaded radius:        {1e3 * RL:.1f} mm")
    print(f"  Effective radius:     {1e3 * RE:.1f} mm")
    print(f"  Vertical deflection:  {1e3 * rho:.1f} mm")

    print("Stiffness")
    print(f"  Longitudinal:         {1e-3 * Cx:.1f} N/mm")
    print(f"  Lateral:              {1e-3 * Cy:.1f} N/mm")
    print(f"  Vertical:             {1e-3 * Cz:.1f} N/mm")

    print("Contact patch dimensions")
    print(f"  Length:               {1e3 * a:.1f} mm")
    print(f"  Width:                {1e3 * b:.1f} mm")
    print(f"  Pneumatic trail:      {1e3 * t:.1f} mm")
