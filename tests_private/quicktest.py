from tyredyn import Tyre
import numpy as np
from pathlib import Path

filepath = Path(r'/tyres_example\car205_60R19.tir')

# initialize tyre
tyre = Tyre(
    filepath        = filepath,
    use_model_type  = 'MF62',
    validate        = False,
    use_alpha_star  = True,
    use_gamma_star  = True,
    use_lmu_star    = True,
    use_turn_slip   = True,
    check_format    = True,
    check_limits    = True,
    use_mfeval_mode = False)

# input state
SA = 3.0 * np.pi / 180
SL = 0.05
FZ = 4500.0
P = 1.8e5
IA = 0.0
VX = 200 / 3.6
PHIT = 0.0

[FX, FY, FZ,
 MX, MY, MZ,
 SL, SA, IA, PHIT, VX, P, N,
 R_omega, RE, rho, RL,
 a, b, t,
 mu_x, mu_y,
 MZR,
 Cx, Cy, Cz,
 KYA, iKYA, KXK, iKXK,
 sigma_x, sigma_y] = tyre.find_full_output(SA=SA, SL=SL, FZ=FZ, VX=VX, P=P, IA=IA, PHIT=PHIT, angle_unit="rad")

# planar force
FH = np.sqrt(FX ** 2 + FY ** 2)
mu_ix = np.abs(FX / (FZ + 1e-12))
mu_iy = np.abs(FY / (FZ + 1e-12))
mu_i  = np.abs(FH / (FZ + 1e-12))

Cz_mfeval = FZ / rho

def radps2rpm(input):
    return input * 60.0 / (2.0 * np.pi)

print("\n=== FULL STATE OUTPUT ===")
print("Input state")
print(f"  Slip angle:           {np.rad2deg(SA):.1f} deg")
print(f"  Slip ratio:           {SL:.2f}")
print(f"  Inclination angle:    {np.rad2deg(IA):.1f} deg")
print(f"  Tyre pressure:        {1e-5 * P:.2f} bar")
print(f"  Turn slip:            {PHIT:.2f} /m")

print("Speed")
print(f"  Longitudinal:         {3.6 * VX:.3f} km/h")
print(f"  Angular:              {radps2rpm(N) :.3f} rpm")

print("Forces")
print(f"  Longitudinal:         {FX:.3f} N")
print(f"  Lateral:              {FY:.3f} N")
print(f"  Planar:               {FH:.3f} N")
print(f"  Vertical:             {FZ:.3f} N")

print("Moments")
print(f"  Overturning:          {MX:.3f} Nm")
print(f"  Rolling resistance:   {MY:.3f} Nm")
print(f"  Self-aligning:        {MZ:.3f} Nm")
print(f"  Residual MZ:          {MZR:.3f} Nm")

print("Gradients")
print(f"  Cornering stiffness:  {np.deg2rad(KYA):.3f} N/deg")
print(f"  Slip stiffness:       {1e-2 * KXK:.3f} N/0.01slip")

print(f"Friction coefficients")
#print(f"  Longitudinal (inst):  {mu_ix:.3f}")
print(f"  Longitudinal:         {mu_x:.3f}")
#print(f"  Lateral (inst):       {mu_iy:.3f}")
print(f"  Lateral:              {mu_y:.3f}")
#print(f"  Planar (inst):        {mu_i:.3f}")

print("Relaxation lengths")
print(f"  Longitudinal:         {1e3 * sigma_x:.3f} mm")
print(f"  Lateral:              {-1e3 * sigma_y:.3f} mm")

print("Radii and deflection")
#print(f"  Free radius:          {1e3 * R_omega:.3f} mm")
print(f"  Loaded radius:        {1e3 * RL:.3f} mm")
print(f"  Effective radius:     {1e3 * RE:.3f} mm")
print(f"  Vertical deflection:  {1e3 * rho:.3f} mm")

print("Stiffness")
print(f"  Longitudinal:         {1e-3 * Cx:.3f} N/mm")
print(f"  Lateral:              {1e-3 * Cy:.3f} N/mm")
print(f"  Vertical:             {1e-3 * Cz:.3f} N/mm")
print(f"  Vertical (MFeval):    {1e-3 * Cz_mfeval:.3f} N/mm")

print("Contact patch dimensions")
print(f"  Length:               {1e3 * a:.3f} mm")
print(f"  Width:                {1e3 * b:.3f} mm")
print(f"  Pneumatic trail:      {1e3 * t:.3f} mm")
