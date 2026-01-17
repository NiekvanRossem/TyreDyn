from tyredyn.types.aliases import SignalLike
from tyredyn.models.magic_formula_6x import MF6xBase

# import helper subsystems

# import subsystems
from tyredyn.subsystems.radius.radius_mf62 import RadiusMF62

from functools import wraps


class MF62(MF6xBase):
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

        # run the initialization from the MF6xBase class
        super().__init__(data, **settings)

    #------------------------------------------------------------------------------------------------------------------#
    # RADIUS AND DEFLECTION

    def _create_radius_model(self):
        """Update the radius module."""
        return RadiusMF62(self)

    @wraps(RadiusMF62._find_radius)
    def find_radius(self, FX: SignalLike, FY: SignalLike, FZ: SignalLike, *, N: SignalLike, P: SignalLike, IA: SignalLike) -> list[SignalLike]:

        # pre-process the input tyres_example
        (_, _, FZ, N, P, _, _, _,
         _) = self.common._process_data(SA=0.0, SL=0.0, FZ=FZ, N=N, P=P, IA=IA, VX=None, PHIT=0.0, angle_unit="rad")

        return self.radius._find_radius(FX=FX, FY=FY, FZ=FZ, N=N, P=P)

