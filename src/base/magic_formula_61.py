from src.utils.misc import inherit_docstring, allowableData

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
    # FORCES

    @inherit_docstring(self.forces.find_fx_pure)
    def find_fx_pure(self,
            SL:  allowableData,
            FZ:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VS:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["deg", "rad"] = "rad") -> allowableData:
        return self.forces.find_fx_pure(SL, FZ, P, IA, VS, PHI, angle_unit)

    @inherit_docstring(self.forces.find_fy_pure)
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

    @inherit_docstring(self.forces.find_fx_combined)
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

    @inherit_docstring(self.forces.find_fy_combined)
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

    def find_mx_pure(self,
            SA:  allowableData,
            FZ:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VCX: allowableData = None,
            VS:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["deg", "rad"] = "rad") -> allowableData:
        self.moments.find_mx_pure(SA, FZ, P, IA, VCX, VS, PHI, angle_unit)

    def find_my_pure(self,
            SL: allowableData,
            FZ: allowableData,
            P:  allowableData = None,
            IA: allowableData = 0.0,
            VX: allowableData = None,
            angle_unit: Literal["deg", "rad"] = "rad") -> allowableData:
        return self.moments.find_my_pure(SL, FZ, P, IA, VX, angle_unit)

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

    def find_my_combined(self,
            SA: allowableData,
            SL: allowableData,
            FZ: allowableData,
            P:  allowableData = None,
            IA: allowableData = 0.0,
            VX: allowableData = None,
            angle_unit: Literal["deg", "rad"] = "rad") -> allowableData:
        return self.moments.find_my_combined(SA, SL, FZ, P, IA, VX, angle_unit)

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
    #