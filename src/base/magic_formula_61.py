from src.utils.misc import allowableData

from src.base.base_tyre import TyreBase

# import helper modules
from src.helpers.common_mf61 import CommonMF61
from src.helpers.corrections import CorrectionsMF61
from src.helpers.normalize import Normalize
from src.modules.contact_patch.contact_patch_mf61 import ContactPatchMF61

# import modules
from src.modules.forces.forces_mf61 import ForcesMF61
from src.modules.gradients.gradients_mf61 import GradientsMF61
from src.modules.radius_and_deflection.radius_mf61 import RadiusMF61
from src.modules.relaxation.relaxation_mf61 import RelaxationMF61
from src.modules.turn_slip.turn_slip import TurnSlip
from src.modules.friction_coefficient.friction_mf61 import FrictionMF61
from src.modules.trail.trail_mf61 import TrailMF61
from src.modules.moments.moments_mf61 import MomentsMF61
from src.modules.contact_patch.contact_patch_mf61 import ContactPatchMF61
from src.modules.radius_and_deflection.radius_mf61 import RadiusMF61
from src.modules.stiffness.stiffness_mf61 import StiffnessMF61

from functools import wraps
from typing import Literal

class MF61(TyreBase):
    """
    Class definition for the Magic Formula 6.1 tyre model. Initialize an instance of this class by calling
    ``Tyre(<filename>.tir)``, where ``<filename>.tir`` is a TIR property file with ``FITTYP`` ``61`` or newer.

    This class contains functions to evaluate the tyre state based on a set of inputs. Equations are mainly based on the
    2012 book by Pacejka & Besselink. Some equations are taken from Besselink's 2010 paper in order to match the TNO
    solver and MFeval. Corrections from Marco Furlan.

    References:
      - Pacejka, H.B. & Besselink, I. (2012). *Tire and Vehicle Dynamics. Third Edition*. Elsevier.
        `doi: 10.1016/c2010-0-68548-8 <https://doi.org/10.1016/c2010-0-68548-8>`_
      - Marco Furlan (2025). *MFeval*. MATLAB Central File Exchange. Retrieved December 18, 2025.
        `mathworks.com/matlabcentral/fileexchange/63618-mfeval <https://mathworks.com/matlabcentral/fileexchange/63618-mfeval>`_
      - International Organization for Standardization (2011). *Road vehicles -- Vehicle dynamics and road-holding
        ability -- Vocabulary* (ISO standard No. 8855:2011)
        `iso.org/obp/ui/#iso:std:iso:8855:ed-2:v1:en <https://www.iso.org/obp/ui/#iso:std:iso:8855:ed-2:v1:en>`_
    """

    def __init__(self, data, **settings):

        # run the initialization from the base class
        super().__init_from_data__(data, **settings)

        # default value if no turn slip is selected
        self.zeta_default = 1.0

        # correction factors to avoid singularities at low speed
        self.eps_x = 1e-6
        self.eps_K = 1e-6
        self.eps_V = 0.1  # set to 0.1 as suggested by Pacejka

        # scaling factor to control decaying friction with increasing speed (set to zero generally)
        self.LMUV = 0.0

        # low friction correction for friction coefficient scaling factor, set to 10 as suggested by Pacejka (4.E8)
        self.A_mu = 10.0

        # import helper functions
        self.correction = CorrectionsMF61(self)
        self.normalize  = Normalize(self)
        self.common     = CommonMF61(self)

        # import modules (order is important here since some modules depend on others)
        self.turn_slip      = TurnSlip(self)            # depends only on helper functions
        self.friction       = FrictionMF61(self)        # depends only on helper functions
        self.stiffness      = StiffnessMF61(self)       # depends only on helper functions
        self.contact_patch  = ContactPatchMF61(self)    # depends on stiffness
        self.radius         = RadiusMF61(self)          # depends on stiffness
        self.trail          = TrailMF61(self)           # depends on turn slip
        self.gradient       = GradientsMF61(self)       # depends on turn slip
        self.relaxation     = RelaxationMF61(self)      # depends on stiffness and gradient
        self.forces         = ForcesMF61(self)          # depends on turn slip, gradient, and forces
        self.moments        = MomentsMF61(self)         # depends on turn slip, friction, trail, gradient, and forces

    #------------------------------------------------------------------------------------------------------------------#
    # STATE OUTPUTS

    def find_forces(self,
            SA:  allowableData,
            SL:  allowableData,
            FZ:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VCX: allowableData = None,
            VS:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["deg", "rad"] = "rad") -> list[allowableData]:
        """
        Finds the force vector for combined slip conditions. Order is ``FX``, ``FY``, ``FZ``.

        Parameters
        ----------
        SA : allowableData
            Slip angle.
        SL : allowableData
            Slip ratio.
        FZ : allowableData
            Vertical load.
        P : allowableData, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : allowableData, optional
            Inclination angle with respect to the ground plane (will default to zero if not specified).
        VCX : allowableData, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).
        VS : allowableData, optional
            Contact patch slip speed (will default to zero if not specified).
        PHI : allowableData, optional
            Turn slip (will default to zero if not specified).
        angle_unit : str, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        FX, FY, FZ : list[allowableData]
            Force vector.
        """

        # find planar forces
        FX = self.forces.find_fx_combined(SA, SL, FZ, P, IA, VCX, VS, angle_unit=angle_unit)
        FY = self.forces.find_fy_combined(SA, SL, FZ, P, IA, VCX, VS, PHI, angle_unit=angle_unit)
        return [FX, FY, FZ]

    def find_moments(self,
            SA:  allowableData,
            SL:  allowableData,
            FZ:  allowableData,
            VX:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VC:  allowableData = None,
            VCX: allowableData = None,
            VS:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["deg", "rad"] = "rad") -> list[allowableData]:
        """
        Finds the moment vector for combined slip conditions. Order is ``MX``, ``MY``, ``MZ``.

        Parameters
        ----------
        SA : allowableData
            Slip angle.
        SL : allowableData
            Slip ratio.
        FZ : allowableData
            Vertical load.
        VX : allowableData, optional
            Wheel centre longitudinal speed (will default to ``VCX`` if not specified, if ``VCX`` is also not specified
            it will default to ``LONGVL``).
        P : allowableData, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : allowableData, optional
            Inclination angle with respect to the ground plane (will default to zero if not specified).
        VC : allowableData, optional
            Contact patch speed (will default to ``LONGVL`` if not specified).
        VCX : allowableData, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).
        VS : allowableData, optional
            Contact patch slip speed (will default to zero if not specified).
        PHI : allowableData, optional
            Turn slip (will default to zero if not specified).
        angle_unit : str, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        MX, MY, MZ : list[allowableData]
            Moment vector.
        """

        VX = VCX if VCX is not None else VX

        MX = self.moments.find_mx_combined(SA, SL, FZ, P, IA, VCX, VS, PHI, angle_unit)
        MY = self.moments.find_my_combined(SA, SL, FZ, P, IA, VX, VS, PHI, angle_unit)
        MZ = self.moments.find_mz_combined(SA, SL, FZ, P, IA, VC, VCX, VS, PHI, angle_unit)
        return [MX, MY, MZ]

    def find_force_moment(self,
            SA:  allowableData,
            SL:  allowableData,
            FZ:  allowableData,
            VX:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VC:  allowableData = None,
            VCX: allowableData = None,
            VS:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["deg", "rad"] = "rad") -> list[allowableData]:
        """
        Finds the total force and moment vector of the tyre for combined slip conditions. Order is ``FX``, ``FY``,
        ``FZ``, ``MX``, ``MY``, ``MZ``.

        Parameters
        ----------
        SA : allowableData
            Slip angle.
        SL : allowableData
            Slip ratio.
        FZ : allowableData
            Vertical load.
        VX : allowableData, optional
            Wheel centre longitudinal speed (will default to ``VCX`` if not specified, if ``VCX`` is also not specified
            it will default to ``LONGVL``).
        P : allowableData, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : allowableData, optional
            Inclination angle with respect to the ground plane (will default to zero if not specified).
        VC : allowableData, optional
            Contact patch speed (will default to ``LONGVL`` if not specified).
        VCX : allowableData, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).
        VS : allowableData, optional
            Contact patch slip speed (will default to zero if not specified).
        PHI : allowableData, optional
            Turn slip (will default to zero if not specified).
        angle_unit : str, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        FX, FY, FZ, MX, MY, MZ : list[allowableData]
            Force and moment vector.
        """

        [FX, FY, FZ] = self.find_forces(SA, SL, FZ, P, IA, VCX, VS, PHI, angle_unit)
        [MX, MY, MZ] = self.find_moments(SA, SL, FZ, VX, P, IA, VC, VCX, VS, PHI, angle_unit)
        return [FX, FY, FZ, MX, MY, MZ]

    def find_lateral_output(self,
            SA:  allowableData,
            FZ:  allowableData,
            N:   allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VC:  allowableData = None,
            VCX: allowableData = None,
            VS:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["rad", "deg"] = "rad") -> list[allowableData]:
        """
        Finds the free rolling outputs commonly used in lateral vehicle models. Order is ``FY``, ``MX``, ``MZ``, ``RL``,
        ``sigma_y``.

        Parameters
        ----------
        SA : allowableData
            Slip angle.
        FZ : allowableData
            Vertical load.
        N : allowableData
            Angular speed of the wheel.
        P : allowableData, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : allowableData, optional
            Inclination angle with respect to the ground plane (will default to zero if not specified).
        VC : allowableData, optional
            Contact patch speed (will default to ``LONGVL`` if not specified).
        VCX : allowableData, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).
        VS : allowableData, optional
            Contact patch slip speed (will default to zero if not specified).
        PHI : allowableData, optional
            Turn slip (will default to zero if not specified).
        angle_unit : str, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        FY, MX, MZ, RL, sigma_y: list[allowableData]
            Lateral output vector.
        """

        FY = self.find_fy_pure(SA, FZ, P, IA, VCX, VS, PHI, angle_unit)
        MX = self.find_mx_pure(SA, FZ, P, IA, VCX, VS, PHI, angle_unit)
        MZ = self.find_mz_pure(SA, FZ, P, IA, VC, VCX, VS, PHI)
        RL = self.find_loaded_radius(0.0, FY, FZ, N, P)
        sigma_y = self.find_lateral_relaxation(FZ, P, IA, PHI, angle_unit)
        return [FY, MX, MZ, RL, sigma_y]

    def find_longitudinal_output(self,
            SL: allowableData,
            FZ: allowableData,
            N:  allowableData,
            P:  allowableData = None,
            IA: allowableData = 0.0,
            VS: allowableData = 0.0,
            angle_unit: Literal["rad", "deg"] = "rad") -> list[allowableData]:
        """
        Finds the output signals commonly used in longitudinal vehicle models. Order is ``FX``, ``MY``, ``RL``, ``RE``,
        ``sigma_x``.

        Parameters
        ----------
        SL : allowableData
            Slip ratio.
        FZ : allowableData
            Vertical load.
        N : allowableData
            Angular speed of the wheel.
        P : allowableData, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : allowableData, optional
            Inclination angle with respect to the ground plane (will default to zero if not specified).
        VS : allowableData, optional
            Contact patch slip speed (will default to zero if not specified).
        angle_unit : str, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        FX, MY, RL, RE, sigma_x: list[allowableData]
            Longitudinal output vector.
        """

        FX = self.forces.find_fx_pure(SL, FZ, P, IA, VS, angle_unit)
        MY = self.forces.find_my_pure(SL, FZ, P, IA, VS, angle_unit)
        RL = self.radius.find_loaded_radius(FX, 0.0, FZ, N, P)
        RE = self.radius.find_effective_radius(FZ, N, P)
        sigma_x = self.relaxation.find_longitudinal_relaxation(FZ, P)
        return [FX, MY, RL, RE, sigma_x]

    def find_full_output(self,
            SA:  allowableData,
            SL:  allowableData,
            FZ:  allowableData,
            VX:  allowableData,
            N:   allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VC:  allowableData = None,
            VCX: allowableData = None,
            VS:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["rad", "deg"] = "rad") -> list[allowableData]:
        """
        Finds the full output state of the tyre. Not recommended to use this in performance-sensitive vehicle
        simulation, as some functions are called multiple times.

        Parameters
        ----------
        SA : allowableData
            Slip angle.
        SL : allowableData
            Slip ratio.
        FZ : allowableData
            Vertical load.
        VX : allowableData
            Wheel centre longitudinal speed.
        N : allowableData
            Angular speed of the wheel.
        P : allowableData, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : allowableData, optional
            Inclination angle with respect to the ground plane (will default to zero if not specified).
        VC : allowableData, optional
            Contact patch speed (will default to ``VX`` if not specified).
        VCX : allowableData, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).
        VS : allowableData, optional
            Contact patch slip speed (will default to zero if not specified).
        PHI : allowableData, optional
            Turn slip (will default to zero if not specified).
        angle_unit : str, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        OUT: list[allowableData]
            Full output state.
        """

        # set default values for optional arguments
        P   = self.INFLPRES if P is None else P
        VC  = self.LONGVL if VC is None else VC
        VCX = self.LONGVL if VCX is None else VCX
        PHI = 0.0 if PHI is None else PHI

        # turn slip correction
        if self._use_turn_slip:
            zeta_0 = 0.0
            zeta_2 = self.turn_slip._find_zeta_2(SA, FZ, PHI)
            zeta_4 = self.turn_slip._find_zeta_4(FZ, P, IA, VCX, VS, PHI, zeta_2, angle_unit)
            zeta_6 = self.turn_slip._find_zeta_6()
            zeta_7 = self.turn_slip._find_zeta_7()
            zeta_8 = self.turn_slip._find_zeta_8(FZ, P, IA, VS, angle_unit)
        else:
            zeta_0 = self.zeta_default
            zeta_2 = self.zeta_default
            zeta_4 = self.zeta_default
            zeta_6 = self.zeta_default
            zeta_7 = self.zeta_default
            zeta_8 = self.zeta_default

        # force and moment vector
        [FX, FY, FZ, MX, MY, MZ] = self.find_force_moment(SA, SL, FZ, VX, P, IA, VC, VCX, VS, PHI, angle_unit)

        # residual self-aligning couple
        MZR = self.moments._mz_main_routine(SA, SL, FZ, P, IA, VC, VCX, VS, PHI,
            zeta_0, zeta_2, zeta_4, zeta_6, zeta_7, zeta_8,
            combined_slip = True,
            angle_unit = angle_unit)

        # free, loaded, and effective radii, and deflection
        R_omega = self.radius.find_free_radius(N) if not self._use_mfeval_mode else None
        RE = self.radius.find_effective_radius(FZ, N, P)
        RL = self.radius.find_loaded_radius(FX, FY, FZ, N, P)
        rho = self.radius.find_deflection(FX, FY, FZ, N, P)

        # pneumatic trail
        t = self.trail.find_trail_combined(SA, SL, FZ, P, IA, VC, VCX, VS, PHI, angle_unit)

        # friction coefficients
        mu_x = self.friction.find_mu_x(FZ, P, IA, VS, angle_unit)
        mu_y = self.friction.find_mu_y(FZ, P, IA, VS, angle_unit)

        # contact patch dimensions
        a, b = self.contact_patch.find_contact_patch(FZ, P)

        # tyre stiffness
        Cx = self.stiffness.find_longitudinal_stiffness(FZ, P)
        Cy = self.stiffness.find_lateral_stiffness(FZ, P)
        Cz = self.stiffness.find_vertical_stiffness(P)

        # slip stiffness
        KXK = self.gradient.find_slip_stiffness(FZ, P)
        KYA = self.gradient.find_cornering_stiffness(FZ, P, IA, PHI, angle_unit)

        # relaxation length
        sigma_x = self.relaxation.find_longitudinal_relaxation(FZ, P)
        sigma_y = self.relaxation.find_lateral_relaxation(FZ, P, IA, PHI, angle_unit)

        # instantaneous slip stiffness
        iKYA = self.gradient.find_instant_kya(SA, FY)
        iKXK = self.gradient.find_instant_kxk(SL, FX) if self._use_mfeval_mode else None

        # assemble final output
        if self._use_mfeval_mode:

            # compatibility mode. Output vector has the same order as MFeval
            output = [FX, FY, FZ, MX, MY, MZ, SL, SA, IA, PHI, VX, P, RE, rho, 2 * a,
                      t, mu_x, mu_y, N, RL, 2 * b, MZR, Cx, Cy, Cz, KYA, sigma_x, sigma_y, iKYA, KXK]
        else:

            # more organized output vector
            output = [
                FX, FY, FZ,                 # FORCES
                MX, MY, MZ,                 # MOMENTS
                SL, SA, IA, PHI, VX, P, N,  # INPUT STATE
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
    def find_fx_pure(self,
            SL:  allowableData,
            FZ:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VS:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["deg", "rad"] = "rad") -> allowableData:
        return self.forces.find_fx_pure(SL, FZ, P, IA, VS, PHI, angle_unit)

    @wraps(ForcesMF61.find_fy_pure)
    def find_fy_pure(self,
            SA:  allowableData,
            FZ:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VCX: allowableData = None,
            VS:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["deg", "rad"] = "rad") -> allowableData:
        return self.forces.find_fy_pure(SA, FZ, P, IA, VCX, VS, PHI, angle_unit)

    @wraps(ForcesMF61.find_fx_combined)
    def find_fx_combined(self,
            SA:  allowableData,
            SL:  allowableData,
            FZ:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VCX: allowableData = None,
            VS:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["deg", "rad"] = "rad") -> allowableData:
        return self.forces.find_fx_combined(SA, SL, FZ, P, IA, VCX, VS, PHI, angle_unit)

    @wraps(ForcesMF61.find_fy_combined)
    def find_fy_combined(self,
            SA:  allowableData,
            SL:  allowableData,
            FZ:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VCX: allowableData = None,
            VS:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["deg", "rad"] = "rad") -> allowableData:
        return self.forces.find_fy_combined(SA, SL, FZ, P, IA, VCX, VS, PHI, angle_unit)

    #------------------------------------------------------------------------------------------------------------------#
    # MOMENTS

    @wraps(MomentsMF61.find_mx_pure)
    def find_mx_pure(self,
            SA:  allowableData,
            FZ:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VCX: allowableData = None,
            VS:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["deg", "rad"] = "rad") -> allowableData:
        return self.moments.find_mx_pure(SA, FZ, P, IA, VCX, VS, PHI, angle_unit)

    @wraps(MomentsMF61.find_my_pure)
    def find_my_pure(self,
            SL: allowableData,
            FZ: allowableData,
            P:  allowableData = None,
            IA: allowableData = 0.0,
            VX: allowableData = None,
            angle_unit: Literal["deg", "rad"] = "rad") -> allowableData:
        return self.moments.find_my_pure(SL, FZ, P, IA, VX, angle_unit)

    @wraps(MomentsMF61.find_mz_pure)
    def find_mz_pure(self,
            SA:  allowableData,
            FZ:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VC:  allowableData = None,
            VCX: allowableData = None,
            VS:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["deg", "rad"] = "rad") -> allowableData:
        return self.moments.find_mz_pure(SA, FZ, P, IA, VC, VCX, VS, PHI, angle_unit)

    @wraps(MomentsMF61.find_mx_combined)
    def find_mx_combined(self,
            SA:  allowableData,
            SL:  allowableData,
            FZ:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VCX: allowableData = None,
            VS:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["deg", "rad"] = "rad") -> allowableData:
        return self.moments.find_mx_combined(SA, SL, FZ, P, IA, VCX, VS, PHI, angle_unit)

    @wraps(MomentsMF61.find_my_combined)
    def find_my_combined(self,
            SA:  allowableData,
            SL:  allowableData,
            FZ:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VX:  allowableData = None,
            VS:  allowableData = 0.0,
            PHI: allowableData = 0.0,
            angle_unit: Literal["deg", "rad"] = "rad") -> allowableData:
        return self.moments.find_my_combined(SA, SL, FZ, P, IA, VX, VS, PHI, angle_unit)

    @wraps(MomentsMF61.find_mz_combined)
    def find_mz_combined(self,
            SA:  allowableData,
            SL:  allowableData,
            FZ:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VC:  allowableData = None,
            VCX: allowableData = None,
            VS:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["deg", "rad"] = "rad") -> allowableData:
        return self.moments.find_mz_combined(SA, SL, FZ, P, IA, VC, VCX, VS, PHI, angle_unit)

    #------------------------------------------------------------------------------------------------------------------#
    # RADIUS AND DEFLECTION

    @wraps(RadiusMF61.find_deflection)
    def find_deflection(self,
            FX: allowableData,
            FY: allowableData,
            FZ: allowableData,
            N:  allowableData,
            P:  allowableData = None) -> allowableData:
        return self.radius.find_deflection(FX, FY, FZ, N, P)

    @wraps(RadiusMF61.find_effective_radius)
    def find_effective_radius(self,
            FZ: allowableData,
            N:  allowableData,
            P:  allowableData = None) -> allowableData:
        return self.radius.find_effective_radius(FZ, N, P)

    @wraps(RadiusMF61.find_free_radius)
    def find_free_radius(self, N: allowableData) -> allowableData:
        return self.radius.find_free_radius(N)

    @wraps(RadiusMF61.find_loaded_radius)
    def find_loaded_radius(self,
            FX: allowableData,
            FY: allowableData,
            FZ: allowableData,
            N:  allowableData,
            P:  allowableData = None) -> allowableData:
        return self.radius.find_loaded_radius(FX, FY, FZ, N, P)

    #------------------------------------------------------------------------------------------------------------------#
    # FRICTION COEFFICIENT

    @wraps(FrictionMF61.find_mu_x)
    def find_mu_x(self,
            FZ: allowableData,
            P:  allowableData = None,
            IA: allowableData = 0.0,
            VS: allowableData = 0.0,
            angle_unit: Literal["rad", "deg"] = "rad") -> allowableData:
        return self.friction.find_mu_x(FZ, P, IA, VS, angle_unit)

    @wraps(FrictionMF61.find_mu_y)
    def find_mu_y(self,
            FZ: allowableData,
            P:  allowableData = None,
            IA: allowableData = 0.0,
            VS: allowableData = 0.0,
            angle_unit: Literal["rad", "deg"] = "rad") -> allowableData:
        return self.friction.find_mu_y(FZ, P, IA, VS, angle_unit)

    #------------------------------------------------------------------------------------------------------------------#
    # TYRE STIFFNESS

    @wraps(StiffnessMF61.find_lateral_stiffness)
    def find_lateral_stiffness(self,
            FZ: allowableData,
            P: allowableData = None) -> allowableData:
        return self.stiffness.find_lateral_stiffness(FZ, P)

    @wraps(StiffnessMF61.find_longitudinal_stiffness)
    def find_longitudinal_stiffness(self,
            FZ: allowableData,
            P: allowableData = None) -> allowableData:
        return self.stiffness.find_longitudinal_stiffness(FZ, P)

    @wraps(StiffnessMF61.find_vertical_stiffness)
    def find_vertical_stiffness(self, P: allowableData) -> allowableData:
        return self.stiffness.find_vertical_stiffness(P)

    #------------------------------------------------------------------------------------------------------------------#
    # CONTACT PATCH DIMENSIONS

    @wraps(ContactPatchMF61.find_contact_patch)
    def find_contact_patch(self, FZ: allowableData, P: allowableData = None) -> list[allowableData]:
        return self.contact_patch.find_contact_patch(FZ, P)

    # ------------------------------------------------------------------------------------------------------------------#
    # PNEUMATIC TRAIL

    @wraps(TrailMF61.find_trail_pure)
    def find_trail_pure(self,
            SA:  allowableData,
            FZ:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VC:  allowableData = None,
            VCX: allowableData = None,
            VS:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["rad", "deg"] = "rad") -> allowableData:
        return self.trail.find_trail_pure(SA, FZ, P, IA, VC, VCX, VS, PHI, angle_unit)

    @wraps(TrailMF61.find_trail_combined)
    def find_trail_combined(self,
            SA:  allowableData,
            SL:  allowableData,
            FZ:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VC:  allowableData = None,
            VCX: allowableData = None,
            VS:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["rad", "deg"] = "rad") -> allowableData:
        return self.trail.find_trail_combined(SA, SL, FZ, P, IA, VC, VCX, VS, PHI, angle_unit)

    #------------------------------------------------------------------------------------------------------------------#
    # GRADIENTS

    @wraps(GradientsMF61.find_cornering_stiffness)
    def find_cornering_stiffness(self,
            FZ:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["rad", "deg"] = "rad") -> allowableData:
        return self.gradient.find_cornering_stiffness(FZ, P, IA, PHI, angle_unit)

    @wraps(GradientsMF61.find_slip_stiffness)
    def find_slip_stiffness(self, FZ: allowableData, P:  allowableData = None) -> allowableData:
        return self.gradient.find_slip_stiffness(FZ, P)

    @wraps(GradientsMF61.find_camber_stiffness)
    def find_camber_stiffness(self, FZ: allowableData, P:  allowableData = None) -> allowableData:
        return self.gradient.find_camber_stiffness(FZ, P)

    @wraps(GradientsMF61.find_instant_kya)
    def find_cornering_stiffness_instant(self, FZ: allowableData, P:  allowableData = None) -> allowableData:
        return self.gradient.find_instant_kya(FZ, P)

    @wraps(GradientsMF61.find_instant_kxk)
    def find_slip_stiffness_instant(self, SL: allowableData, FX: allowableData) -> allowableData:
        return self.gradient.find_instant_kxk(SL, FX)

    #------------------------------------------------------------------------------------------------------------------#
    # RELAXATION LENGTHS

    @wraps(RelaxationMF61.find_lateral_relaxation)
    def find_lateral_relaxation(self,
            FZ:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["rad", "deg"] = "rad") -> allowableData:
        return self.relaxation.find_lateral_relaxation(FZ, P, IA, PHI, angle_unit)

    @wraps(RelaxationMF61.find_longitudinal_relaxation)
    def find_longitudinal_relaxation(self,
            FZ: allowableData,
            P:  allowableData = None) -> allowableData:
        return self.relaxation.find_longitudinal_relaxation(FZ, P)
