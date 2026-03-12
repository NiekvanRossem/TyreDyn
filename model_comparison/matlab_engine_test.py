from model_comparison.utils.process_output import process_mfeval
from tyredyn.infrastructure.paths import PROJECT_ROOT
import numpy as np
import matlab.engine
import sys

# check if you have the 64-bit version of Python (this is required for matlab)
assert sys.maxsize > 2**32, "You need a 64 bit version of Python"

# start matlab
eng = matlab.engine.start_matlab()

# add MFeval to path
mfeval_path = r"C:\Users\niekv\Documents\MATLAB\Toolboxes\MFeval"
eng.cd(mfeval_path, nargout=0)

# input state
SA = np.deg2rad(3.0)
SL = 0.05
FZ = 4500.0
P = 1.8e5
IA = -np.deg2rad(0.4)
VX = 200 / 3.6
PHIT = 0.1

# convert to Matlab array
inputs = [FZ, SL, SA, IA, PHIT, VX, P]
inputs = matlab.double(inputs)

# set use mode
usemode = matlab.double(122)

# path to TIR file
tir_file = str(PROJECT_ROOT / 'tyres_example' / 'car205_60R19.tir')

out = eng.mfeval(tir_file, inputs, usemode, nargout=1)
process_mfeval(out)