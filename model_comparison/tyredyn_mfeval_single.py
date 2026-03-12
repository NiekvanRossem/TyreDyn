from tyredyn.infrastructure.paths import PROJECT_ROOT
from model_comparison.utils.table_plot import print_table_section
from model_comparison.utils.unit_conversion import si2display
from model_comparison.utils.process_output import process_mfeval, process_tyredyn, plot_comparison
import numpy as np
import sys

#----------------------------------------------------------------------------------------------------------------------#
# SETUP

# choose whether to use turn slip
use_turn_slip = False

# check if you have the 64-bit version of Python (this is required for the matlab engine)
assert sys.maxsize > 2**32, "You need a 64 bit version of Python"

# path to TIR file (change this for your case)
tir_file = PROJECT_ROOT / 'tyres_example' / 'car205_60R19.tir'

#----------------------------------------------------------------------------------------------------------------------#
# INITIALIZE MFEVAL

# import and start matlab engine
import matlab.engine
eng = matlab.engine.start_matlab()

# add MFeval to path (change this for your case)
mfeval_path = r"C:\Users\niekv\Documents\MATLAB\Toolboxes\MFeval"
eng.cd(mfeval_path, nargout=0)

#----------------------------------------------------------------------------------------------------------------------#
# INITIALIZE TYREDYN

# import library
from tyredyn import Tyre

# create tyre instance
tyredyn_tyre = Tyre(
    filepath        = tir_file,
    validate        = False,
    use_alpha_star  = True,
    use_gamma_star  = True,
    use_lmu_star    = True,
    use_turn_slip   = use_turn_slip,
    check_format    = True,
    check_limits    = True,
    use_mfeval_mode = False
)

#----------------------------------------------------------------------------------------------------------------------#
# PREPARE INPUT

# input state
SA   = np.deg2rad(7.0)  # slip angle
SL   = 0.00             # slip ratio
FZ   = 4500.0           # vertical load
P    = 1.8e5            # pressure
IA   = -np.deg2rad(1.1) # inclination angle
VX   = 200 / 3.6        # speed
PHIT = 0.0              # turn slip (will be ignored if use_turn_slip is set to False)

# store in a Matlab array for MFeval
inputs_mfeval = matlab.double([FZ, SL, SA, IA, PHIT, VX, P])

# set use mode for MFeval to match TyreDyn
mfeval_usemode = matlab.double(122 if use_turn_slip else 121)

#----------------------------------------------------------------------------------------------------------------------#
# READ TYRE STATE

# MFeval
out_mfeval = eng.mfeval(str(tir_file), inputs_mfeval, mfeval_usemode, nargout=1)

# TyreDyn
out_tyredyn = tyredyn_tyre.find_full_output(SA=SA, SL=SL, FZ=FZ, VX=VX, P=P, IA=IA, PHIT=PHIT, angle_unit="rad")

# print results
plot_comparison(out_tyredyn, out_mfeval)
