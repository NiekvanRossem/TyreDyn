from src.helpers.common_mf61 import CommonMF61
from src.helpers.corrections import CorrectionsMF61
from src.helpers.normalize import Normalize

from src.modules.forces.forces_mf61 import ForcesMF61
from src.modules.gradients.gradients_mf61 import GradientsMF61
from src.modules.turn_slip.turn_slip import TurnSlip
from src.modules.friction_coefficient.friction_mf61 import FrictionMF61

from src.base.base_tyre import TyreBase

class MF61new(TyreBase):
    """
        Class definition for the Magic Formula 6.1 tyre model. Initialize an instance of this class by calling
        ``Tyre(<filename.tir>)``, where ``<filename.tir>`` is a TIR property file with ``FITTYP`` ``61`` or newer.

        This class contains functions to evaluate the tyre state based on a set of inputs.

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

        # turn slip correction factors if turn slip is disabled
        self.zeta_0_default = 1.0
        self.zeta_1_default = 1.0
        self.zeta_2_default = 1.0
        self.zeta_3_default = 1.0
        self.zeta_4_default = 1.0
        self.zeta_5_default = 1.0
        self.zeta_6_default = 1.0
        self.zeta_7_default = 1.0
        self.zeta_8_default = 1.0

        # correction factors to avoid singularities at low speed
        self.eps_x = 1e-6
        self.eps_K = 1e-6
        self.eps_V = 0.1  # set to 0.1 as suggested by Pacejka

        # scaling factor to control decaying friction with increasing speed (set to zero generally)
        self.LMUV = 0.0

        # low friction correction for friction coefficient scaling factor, set to 10 as suggested by Pacejka (4.E8)
        self.A_mu = 10.0

        # import helper functions
        self.corrections    = CorrectionsMF61(self)
        self.normalize      = Normalize(self)

        # import modules
        self.forces         = ForcesMF61(self)
        self.moments        = MomentsMF61(self)
        self.friction_coeff = FrictionMF61(self)
        self.gradients      = GradientsMF61(self)
        self.turn_slip      = TurnSlip(self)
