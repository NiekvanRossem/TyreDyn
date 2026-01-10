from src.utils.formatting import SignalLike, AngleUnit
from src.tyre_models.base_tyre import TyreBase

# import helper modules
from src.helpers.common_mf6x import CommonMF6x
from src.helpers.corrections_mf6x import CorrectionsMF6x
from src.helpers.normalize import Normalize

# import modules
from src.modules.contact_patch.contact_patch_mf6x import ContactPatchMF6x
from src.modules.forces.forces_mf6x import ForcesMF6x
from src.modules.friction_coefficient.friction_mf6x import FrictionMF6x
from src.modules.gradients.gradients_mf6x import GradientsMF6x
from src.modules.moments.moments_mf6x import MomentsMF6x
from src.modules.relaxation.relaxation_mf6x import RelaxationMF6x
from src.modules.trail.trail_mf6x import TrailMF6x
from src.modules.turn_slip.turn_slip_mf6x import TurnSlipMF6x
from src.modules.stiffness.stiffness_mf6x import StiffnessMF6x

from functools import wraps

class MF6xBase(TyreBase):
    """
    Class definition for the Magic Formula 6.1 and 6.2 tyre models. This class contains all the shared modules between
    these models. Do not use this class directly, but use ``MF61`` or ``MF62`` instead.

    This class contains functions to evaluate the tyre state based on a set of inputs. Equations are mainly based on the
    2012 book by Pacejka & Besselink. Some equations are taken from Besselink's 2010 paper in order to match the TNO
    solver and MFeval. Corrections from Marco Furlan.

    References:
      - Pacejka, H.B. & Besselink, I. (2012). *Tire and Vehicle Dynamics. Third Edition*. Elsevier.
        `doi: 10.1016/c2010-0-68548-8 <https://doi.org/10.1016/c2010-0-68548-8>`_
      - Besselink, I.J.M. & Schmeitz, A.J.C. & Pacejka, H.B. (2010). *An improved Magic Formula/Swift tyre model that
        can handle inflation pressure changes*. Vehicle System Dynamics, 48(sup1), 337–352.
        `doi: 10.1080/00423111003748088 <https://doi-org.tudelft.idm.oclc.org/10.1080/00423111003748088>`_
      - Besselink, I.J.M. & Schmeitz, A.J.C. & Pacejka, H.B. (2010). *An improved Magic Formula/Swift tyre model that
        can handle inflation pressure changes* **[Unpublished manuscript]**. Retrieved 30 December 2025.
        `https://pure.tue.nl/ws/files/3139488/677330157969510.pdf <https://pure.tue.nl/ws/files/3139488/677330157969510.pdf>`_
      - Marco Furlan (2025). *MFeval*. MATLAB Central File Exchange. Retrieved December 18, 2025.
        `mathworks.com/matlabcentral/fileexchange/63618-mfeval <https://mathworks.com/matlabcentral/fileexchange/63618-mfeval>`_
      - International Organization for Standardization (2011). *Road vehicles -- Vehicle dynamics and road-holding
        ability -- Vocabulary* (ISO standard No. 8855:2011)
        `iso.org/obp/ui/#iso:std:iso:8855:ed-2:v1:en <https://www.iso.org/obp/ui/#iso:std:iso:8855:ed-2:v1:en>`_
    """

    def __init__(self, data, **settings):

        # run the initialization from the tyre_models class
        super().__init_from_data__(data, **settings)

        # default value if no turn slip is selected
        self._zeta_default = 1.0

        # correction factors to avoid singularities at low speed
        self._eps_r = 1e-6
        self._eps_x = 1e-6
        self._eps_kappa = 1e-6
        self._eps_V = 0.1  # set to 0.1 as suggested by Pacejka

        # scaling factor to control decaying friction with increasing speed (set to zero generally)
        self._LMUV = 0.0

        # low friction correction for friction coefficient scaling factor, The book by Pacejka & Besselink recommends
        # setting this value to 10.0 (4.E8), but a value of 1.0 matches the TNO solver (via Marco Furlan)
        self._A_mu = 1.0

        # import helper functions
        self.normalize = Normalize(self)
        self.correction = CorrectionsMF6x(self)  # depends on _normalize
        self.common = CommonMF6x(self)  # depends on _normalize and correction

        # import modules (order is important here since some modules depend on others)
        self.friction = FrictionMF6x(self)  # depends only on helper functions
        self.stiffness = StiffnessMF6x(self)  # depends only on helper functions
        self.turn_slip = TurnSlipMF6x(self)  # depends on friction and gradients
        self.contact_patch = ContactPatchMF6x(self)  # depends on stiffness
        self.radius = self._create_radius_model()
        self.trail = TrailMF6x(self)  # depends on turn slip
        self.gradient = GradientsMF6x(self)  # depends on turn slip
        self.relaxation = RelaxationMF6x(self)  # depends on stiffness and gradient
        self.forces = ForcesMF6x(self)  # depends on turn slip, gradient, and forces
        self.moments = MomentsMF6x(self)  # depends on turn slip, friction, trail, gradient, and forces

    # ------------------------------------------------------------------------------------------------------------------#
    # STATE OUTPUTS

    def find_forces(
            self,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            *,
            N:    SignalLike = None,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> list[SignalLike]:
        """
        Finds the force vector for combined slip conditions. Order is ``FX``, ``FY``, ``FZ``.

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
        angle_unit : str, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        FX, FY, FZ : list[SignalLike]
            Force vector.
        """

        # pre-process the input example_tyres
        (SA, SL, FZ, N, P, IA, VX, PHIT,
         angle_unit) = self.common._preprocess_data(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT, angle_unit=angle_unit)

        # find planar forces
        FX = self.forces._find_fx_combined(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX)
        FY = self.forces._find_fy_combined(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)
        return [FX, FY, FZ]

    def find_moments(
            self,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            *,
            N:    SignalLike = None,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
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
        angle_unit : str, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        MX, MY, MZ : list[SignalLike]
            Moment vector.
        """

        # pre-process the input example_tyres
        (SA, SL, FZ, N, P, IA, VX, PHIT,
         angle_unit) = self.common._preprocess_data(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT,
                                                    angle_unit=angle_unit)

        MX = self.moments._find_mx_combined(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)
        MY = self.moments._find_my_combined(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)
        MZ = self.moments._find_mz_combined(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)
        return [MX, MY, MZ]

    def find_force_moment(
            self,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            *,
            N:    SignalLike = None,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
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
        angle_unit : str, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        FX, FY, FZ, MX, MY, MZ : list[SignalLike]
            Force and moment vector.
        """

        # pre-process the input example_tyres
        (SA, SL, FZ, N, P, IA, VX, PHIT,
         angle_unit) = self.common._preprocess_data(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT,
                                                    angle_unit=angle_unit)

        # find planar forces
        FX = self.forces._find_fx_combined(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX)
        FY = self.forces._find_fy_combined(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

        # find moments
        MX = self.moments._find_mx_combined(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)
        MY = self.moments._find_my_combined(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)
        MZ = self.moments._find_mz_combined(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)
        return [FX, FY, FZ, MX, MY, MZ]

    def find_lateral_output(
            self,
            SA:   SignalLike,
            FZ:   SignalLike,
            *,
            N:    SignalLike = None,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
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
        angle_unit : str, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        FY, MX, MZ, RL, sigma_y: list[SignalLike]
            Lateral output vector.
        """

        # pre-process the input example_tyres
        (SA, _, FZ, N, P, IA, VX, PHIT,
         angle_unit) = self.common._preprocess_data(SA=SA, SL=0.0, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT, angle_unit=angle_unit)

        FY = self.forces._find_fy_pure(SA=SA, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)
        MX = self.moments._find_mx_pure(SA=SA, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)
        MZ = self.moments._find_mz_pure(SA=SA, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)
        _, _, RL, _ = self.radius._find_radius(FX=0.0, FY=FY, FZ=FZ, N=N, P=P, maxiter=maxiter, tolx=tolx)
        sigma_y = self.relaxation._find_lateral_relaxation(SA=SA, SL=0.0, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)
        return [FY, MX, MZ, RL, sigma_y]

    def find_longitudinal_output(
            self,
            SL: SignalLike,
            FZ: SignalLike,
            *,
            N:  SignalLike = None,
            P:  SignalLike = None,
            IA: SignalLike = 0.0,
            VX: SignalLike = None,
            angle_unit: AngleUnit = "rad"
    ) -> list[SignalLike]:
        """
        Finds the output signals commonly used in longitudinal vehicle models. Order is ``FX``, ``MY``, ``RL``,
        ``RE``, ``sigma_x``.

        Parameters
        ----------
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
        angle_unit : str, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        FX, MY, RL, RE, sigma_x : list[SignalLike]
            Longitudinal output vector.
        """

        # pre-process the input example_tyres
        (_, SL, FZ, N, P, IA, VX, PHIT,
         angle_unit) = self.common._preprocess_data(SA=0.0, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=0.0, angle_unit=angle_unit)

        FX = self.forces._find_fx_pure(SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=0.0)
        MY = self.moments._find_my_pure(SL=SL, FZ=FZ, P=P, IA=IA, VX=VX)
        _, RE, RL, _ = self.radius._find_radius(FX=FX, FY=0.0, FZ=FZ, N=N, P=P, maxiter=maxiter, tolx=tolx)
        sigma_x = self.relaxation._find_longitudinal_relaxation(FZ=FZ, P=P)
        return [FX, MY, RL, RE, sigma_x]

    def find_full_output(
            self,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            *,
            N:    SignalLike = None,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad",
            maxiter: int = 30,
            tolx: float = 1e-6
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
        angle_unit : str, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        OUT: list[SignalLike]
            Full output state.
        """

        # pre-process the input example_tyres
        (SA, SL, FZ, N, P, IA, VX, PHIT,
         angle_unit) = self.common._preprocess_data(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT,
                                                    angle_unit=angle_unit)

        # find other velocity components
        VS, VC = self.normalize._find_speeds(SA=SA, SL=SL, VX=VX)
        VCX = VX

        # turn slip correction
        if self._use_turn_slip:
            PHI = self.correction._find_phi(FZ=FZ, N=N, VC=VC, IA=IA, PHIT=PHIT)
            zeta_0 = 0.0
            zeta_2 = self.turn_slip._find_zeta_2(SA=SA, FZ=FZ, PHI=PHI)
            zeta_4 = self.turn_slip._find_zeta_4(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VCX=VCX, VS=VS, PHI=PHI, zeta_2=zeta_2)
            zeta_6 = self.turn_slip._find_zeta_6(PHI)
            zeta_7 = self.turn_slip._find_zeta_7(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VX=VX, VCX=VCX, PHI=PHI, PHIT=PHIT)
            zeta_8 = self.turn_slip._find_zeta_8(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VX=VX, PHIT=PHIT)
        else:
            zeta_0, zeta_2, zeta_4, zeta_6, zeta_7, zeta_8 = 6 * [self._zeta_default]

        # find planar forces
        FX = self.forces._find_fx_combined(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX)
        FY = self.forces._find_fy_combined(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

        # find moments
        MX = self.moments._find_mx_combined(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)
        MY = self.moments._find_my_combined(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)
        MZ = self.moments._find_mz_combined(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

        # residual self-aligning couple
        MZR = self.moments._mz_main_routine(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VX=VX, VC=VC, VCX=VCX, VS=VS, N=N,
                                            zeta_0=zeta_0, zeta_2=zeta_2, zeta_4=zeta_4, zeta_6=zeta_6, zeta_7=zeta_7,
                                            zeta_8=zeta_8, combined_slip=True)

        R_omega, RE, RL, rho = self.radius._find_radius(FX=FX, FY=FY, FZ=FZ, N=N, P=P, maxiter=maxiter, tolx=tolx)

        # pneumatic trail
        t = self.trail.find_trail_combined(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

        # friction coefficients
        mu_x = self.friction._find_mu_x(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VX=VX)
        mu_y = self.friction._find_mu_y(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VX=VX)

        # contact patch dimensions
        a, b = self.contact_patch._find_contact_patch(FZ=FZ, P=P)

        # tyre stiffness
        Cx = self.stiffness._find_longitudinal_stiffness(FZ=FZ, P=P)
        Cy = self.stiffness._find_lateral_stiffness(FZ=FZ, P=P)
        Cz = self.stiffness._find_vertical_stiffness(P=P)

        # slip stiffness
        KXK = self.gradient._find_slip_stiffness(FZ=FZ, P=P)
        KYA = self.gradient._find_cornering_stiffness(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

        # relaxation length
        sigma_x = self.relaxation._find_longitudinal_relaxation(FZ=FZ, P=P)
        sigma_y = self.relaxation._find_lateral_relaxation(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

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
                SL, SA, IA, PHIT, VX, P, N, # INPUT STATE
                R_omega, RE, rho, RL,       # RADII
                2 * a, 2 * b,               # CONTACT PATCH
                t,                          # TRAIL
                mu_x, mu_y,                 # FRICTION COEFFICIENT
                MZR,                        # RESIDUAL MOMENT
                Cx, Cy, Cz,                 # TYRE STIFFNESS
                KYA, iKYA, KXK, iKXK,       # SLIP STIFFNESS
                sigma_x, sigma_y            # RELAXATION LENGTHS
            ]
        return output

    # ------------------------------------------------------------------------------------------------------------------#
    # FORCES

    @wraps(ForcesMF6x._find_fx_pure)
    def find_fx_pure(
            self,
            SL:   SignalLike,
            FZ:   SignalLike,
            *,
            N:    SignalLike = None,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:

        # pre-process the input example_tyres
        (_, SL, FZ, N, P, IA, VX, PHIT,
         angle_unit) = self.common._preprocess_data(SA=0.0, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT,
                                                    angle_unit=angle_unit)

        return self.forces._find_fx_pure(SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

    @wraps(ForcesMF6x._find_fy_pure)
    def find_fy_pure(
            self,
            SA:   SignalLike,
            FZ:   SignalLike,
            *,
            N:    SignalLike = None,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:

        # pre-process the input example_tyres
        (SA, SL, FZ, N, P, IA, VX, PHIT,
         angle_unit) = self.common._preprocess_data(SA=SA, SL=0.0, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT,
                                                    angle_unit=angle_unit)

        return self.forces._find_fy_pure(SA=SA, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

    @wraps(ForcesMF6x._find_fx_combined)
    def find_fx_combined(
            self,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            *,
            N:    SignalLike = None,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:

        # pre-process the input example_tyres
        (SA, SL, FZ, N, P, IA, VX, PHIT,
         angle_unit) = self.common._preprocess_data(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT,
                                                    angle_unit=angle_unit)

        return self.forces._find_fx_combined(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

    @wraps(ForcesMF6x._find_fy_combined)
    def find_fy_combined(
            self,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            *,
            N:    SignalLike = None,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:

        # pre-process the input example_tyres
        (SA, SL, FZ, N, P, IA, VX, PHIT,
         angle_unit) = self.common._preprocess_data(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT,
                                                    angle_unit=angle_unit)

        return self.forces._find_fy_combined(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

    # ------------------------------------------------------------------------------------------------------------------#
    # MOMENTS

    @wraps(MomentsMF6x._find_mx_pure)
    def find_mx_pure(
            self,
            SA:   SignalLike,
            FZ:   SignalLike,
            *,
            N:    SignalLike = None,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:

        # pre-process the input example_tyres
        (SA, _, FZ, N, P, IA, VX, PHIT,
         angle_unit) = self.common._preprocess_data(SA=SA, SL=0.0, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT,
                                                    angle_unit=angle_unit)

        return self.moments._find_mx_pure(SA=SA, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

    @wraps(MomentsMF6x._find_my_pure)
    def find_my_pure(
            self,
            SL: SignalLike,
            FZ: SignalLike,
            *,
            P:  SignalLike = None,
            IA: SignalLike = 0.0,
            VX: SignalLike = None,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:

        # pre-process the input example_tyres
        (_, SL, FZ, _, P, IA, VX, PHIT,
         angle_unit) = self.common._preprocess_data(SA=0.0, SL=SL, FZ=FZ, N=0.0, P=P, IA=IA, VX=VX, PHIT=0.0,
                                                    angle_unit=angle_unit)

        return self.moments._find_my_pure(SL=SL, FZ=FZ, P=P, IA=IA, VX=VX)

    @wraps(MomentsMF6x._find_mz_pure)
    def find_mz_pure(
            self,
            SA:   SignalLike,
            FZ:   SignalLike,
            *,
            N:    SignalLike = None,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:

        # pre-process the input example_tyres
        (SA, _, FZ, N, P, IA, VX, PHIT,
         angle_unit) = self.common._preprocess_data(SA=SA, SL=0.0, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT,
                                                    angle_unit=angle_unit)

        return self.moments._find_mz_pure(SA=SA, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

    @wraps(MomentsMF6x._find_mx_combined)
    def find_mx_combined(
            self,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            *,
            N:    SignalLike = None,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:

        # pre-process the input example_tyres
        (SA, SL, FZ, N, P, IA, VX, PHIT,
         angle_unit) = self.common._preprocess_data(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT,
                                                    angle_unit=angle_unit)

        return self.moments._find_mx_combined(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

    @wraps(MomentsMF6x._find_my_combined)
    def find_my_combined(
            self,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            *,
            N:    SignalLike = None,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:

        # pre-process the input example_tyres
        (SA, SL, FZ, N, P, IA, VX, PHIT,
         angle_unit) = self.common._preprocess_data(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT,
                                                    angle_unit=angle_unit)

        return self.moments._find_my_combined(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

    @wraps(MomentsMF6x._find_mz_combined)
    def find_mz_combined(
            self,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            *,
            N:    SignalLike = None,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:

        # pre-process the input example_tyres
        (SA, SL, FZ, N, P, IA, VX, PHIT,
         angle_unit) = self.common._preprocess_data(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT,
                                                    angle_unit=angle_unit)

        return self.moments._find_mz_combined(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

    # ------------------------------------------------------------------------------------------------------------------#
    # RADIUS AND DEFLECTION

    def _create_radius_model(self):
        raise NotImplementedError

    def find_radius(
            self,
            FX: SignalLike,
            FY: SignalLike,
            FZ: SignalLike,
            *,
            N:  SignalLike,
            P:  SignalLike,
            **kwargs):
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------#
    # FRICTION COEFFICIENT

    @wraps(FrictionMF6x._find_mu_x)
    def find_mu_x(
            self,
            SA: SignalLike,
            SL: SignalLike,
            FZ: SignalLike,
            *,
            P:  SignalLike = None,
            IA: SignalLike = 0.0,
            VX: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:

        # pre-process the input example_tyres
        (SA, SL, FZ, _, P, IA, VX, _,
         angle_unit) = self.common._preprocess_data(SA=SA, SL=SL, FZ=FZ, N=0.0, P=P, IA=IA, VX=VX, PHIT=0.0,
                                                    angle_unit=angle_unit)

        return self.friction._find_mu_x(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VX=VX)

    @wraps(FrictionMF6x._find_mu_y)
    def find_mu_y(
            self,
            SA: SignalLike,
            SL: SignalLike,
            FZ: SignalLike,
            *,
            P:  SignalLike = None,
            IA: SignalLike = 0.0,
            VX: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:

        # pre-process the input example_tyres
        (SA, SL, FZ, _, P, IA, VX, _,
         angle_unit) = self.common._preprocess_data(SA=SA, SL=SL, FZ=FZ, N=0.0, P=P, IA=IA, VX=VX, PHIT=0.0,
                                                    angle_unit=angle_unit)

        return self.friction._find_mu_y(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VX=VX)

    # ------------------------------------------------------------------------------------------------------------------#
    # TYRE STIFFNESS

    @wraps(StiffnessMF6x._find_lateral_stiffness)
    def find_lateral_stiffness(
            self,
            FZ: SignalLike,
            *,
            P:  SignalLike = None
    ) -> SignalLike:

        # pre-process the input example_tyres
        (_, _, FZ, _, P, _, _, _,
         angle_unit) = self.common._preprocess_data(SA=0.0, SL=0.0, FZ=FZ, N=0.0, P=P, IA=0.0, VX=None, PHIT=0.0,
                                                    angle_unit="rad")

        return self.stiffness._find_lateral_stiffness(FZ=FZ, P=P)

    @wraps(StiffnessMF6x._find_longitudinal_stiffness)
    def find_longitudinal_stiffness(
            self,
            FZ: SignalLike,
            *,
            P:  SignalLike = None
    ) -> SignalLike:

        # pre-process the input example_tyres
        (_, _, FZ, _, P, _, _, _,
         _) = self.common._preprocess_data(SA=0.0, SL=0.0, FZ=FZ, N=0.0, P=P, IA=0.0, VX=None, PHIT=0.0,
                                           angle_unit="rad")

        return self.stiffness._find_longitudinal_stiffness(FZ=FZ, P=P)

    @wraps(StiffnessMF6x._find_vertical_stiffness)
    def find_vertical_stiffness(
            self,
            P: SignalLike
    ) -> SignalLike:

        # pre-process the input example_tyres
        (_, _, _, _, P, _, _, _,
         _) = self.common._preprocess_data(SA=0.0, SL=0.0, FZ=self.FNOMIN, N=0.0, P=P, IA=0.0, VX=self.LONGVL, PHIT=0.0,
                                           angle_unit="rad")

        return self.stiffness._find_vertical_stiffness(P)

    # ------------------------------------------------------------------------------------------------------------------#
    # CONTACT PATCH DIMENSIONS

    @wraps(ContactPatchMF6x._find_contact_patch)
    def find_contact_patch(
            self,
            FZ: SignalLike,
            *,
            P:  SignalLike = None
    ) -> list[SignalLike]:

        # pre-process the input example_tyres
        (_, _, FZ, _, P, _, _, _,
         _) = self.common._preprocess_data(SA=0.0, SL=0.0, FZ=FZ, N=0.0, P=P, IA=0.0, VX=None, PHIT=0.0,
                                           angle_unit="rad")

        return self.contact_patch._find_contact_patch(FZ=FZ, P=P)

    # ------------------------------------------------------------------------------------------------------------------#
    # PNEUMATIC TRAIL

    @wraps(TrailMF6x.find_trail_pure)
    def find_trail_pure(
            self,
            SA:   SignalLike,
            FZ:   SignalLike,
            *,
            N:    SignalLike = None,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:

        # pre-process the input example_tyres
        (SA, _, FZ, N, P, IA, VX, PHIT,
         angle_unit) = self.common._preprocess_data(SA=SA, SL=0.0, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT,
                                                    angle_unit=angle_unit)

        return self.trail.find_trail_pure(SA=SA, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

    @wraps(TrailMF6x.find_trail_combined)
    def find_trail_combined(
            self,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            *,
            N:    SignalLike = None,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:

        # pre-process the input example_tyres
        (SA, SL, FZ, N, P, IA, VX, PHIT,
         angle_unit) = self.common._preprocess_data(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT,
                                                    angle_unit=angle_unit)

        return self.trail.find_trail_combined(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

    # ------------------------------------------------------------------------------------------------------------------#
    # GRADIENTS

    @wraps(GradientsMF6x._find_cornering_stiffness)
    def find_cornering_stiffness(
            self,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            *,
            N:    SignalLike = None,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:

        # pre-process the input example_tyres
        (SA, SL, FZ, N, P, IA, VX, PHIT,
         angle_unit) = self.common._preprocess_data(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT,
                                                    angle_unit=angle_unit)

        return self.gradient._find_cornering_stiffness(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

    @wraps(GradientsMF6x._find_slip_stiffness)
    def find_slip_stiffness(
            self,
            FZ: SignalLike,
            *,
            P:  SignalLike = None
    ) -> SignalLike:

        # pre-process the input example_tyres
        (_, _, FZ, _, P, _, _, _,
         _) = self.common._preprocess_data(SA=0.0, SL=0.0, FZ=FZ, N=0.0, P=P, IA=0.0, VX=None, PHIT=0.0,
                                           angle_unit="rad")

        return self.gradient._find_slip_stiffness(FZ=FZ, P=P)

    @wraps(GradientsMF6x._find_camber_stiffness)
    def find_camber_stiffness(
            self,
            FZ: SignalLike,
            *,
            P:  SignalLike = None
    ) -> SignalLike:

        # pre-process the input example_tyres
        (_, _, FZ, _, P, _, _, _,
         _) = self.common._preprocess_data(SA=0.0, SL=0.0, FZ=FZ, N=0.0, P=P, IA=0.0, VX=None, PHIT=0.0,
                                           angle_unit="rad")

        return self.gradient._find_camber_stiffness(FZ=FZ, P=P)

    @wraps(GradientsMF6x.find_instant_kya)
    def find_cornering_stiffness_instant(
            self,
            SA: SignalLike,
            *,
            FY: SignalLike = None,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:

        # pre-process the input example_tyres
        (SA, _, _, _, _, _, _, _,
         angle_unit) = self.common._preprocess_data(SA=SA, SL=0.0, FZ=self.FNOMIN, N=0.0, P=self.INFLPRES, IA=0.0,
                                                    VX=None, PHIT=0.0, angle_unit=angle_unit)

        return self.gradient.find_instant_kya(SA=SA, FY=FY)

    @wraps(GradientsMF6x.find_instant_kxk)
    def find_slip_stiffness_instant(
            self,
            SL: SignalLike,
            FX: SignalLike
    ) -> SignalLike:

        # pre-process the input example_tyres
        (_, SL, _, _, _, _, _, _,
         _) = self.common._preprocess_data(SA=0.0, SL=SL, FZ=self.FNOMIN, N=0.0, P=self.INFLPRES, IA=0.0, VX=None,
                                           PHIT=0.0, angle_unit="rad")

        return self.gradient.find_instant_kxk(SL=SL, FX=FX)

    # ------------------------------------------------------------------------------------------------------------------#
    # RELAXATION LENGTHS

    @wraps(RelaxationMF6x._find_lateral_relaxation)
    def find_lateral_relaxation(
            self,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            *,
            N:    SignalLike = None,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:

        # pre-process the input example_tyres
        (SA, SL, FZ, N, P, IA, VX, PHIT,
         angle_unit) = self.common._preprocess_data(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT,
                                                    angle_unit=angle_unit)

        return self.relaxation._find_lateral_relaxation(SA=SA, SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

    @wraps(RelaxationMF6x._find_longitudinal_relaxation)
    def find_longitudinal_relaxation(
            self,
            FZ: SignalLike,
            *,
            P:  SignalLike = None
    ) -> SignalLike:

        # pre-process the input example_tyres
        (_, _, FZ, _, P, _, _, _,
         _) = self.common._preprocess_data(SA=0.0, SL=0.0, FZ=FZ, N=0.0, P=P, IA=0.0, VX=self.LONGVL, PHIT=0.0,
                                           angle_unit="rad")

        return self.relaxation._find_longitudinal_relaxation(FZ=FZ, P=P)
