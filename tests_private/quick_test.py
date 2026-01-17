import numpy as np
from src.initialize_tyre import Tyre
import matplotlib.pyplot as plt
from pathlib import Path

filepath = Path(r"C:\Users\niekv\Documents\5. Personal engineering scripts\TyreDyn\example_tyres\car205_60R19.tir")

tyre = Tyre(
    filepath        = filepath,
    validate        = False,
    use_alpha_star  = True,
    use_gamma_star  = True,
    use_lmu_star    = True,
    use_turn_slip   = True,
    use_model_type  = "MF62",
    check_format    = True,
    check_limits    = True,
    use_mfeval_mode = True
)
# input state
SA   = 16.5
SL   = 0.55
FZ   = 4500
P    = 1.8e5
IA   = 0.0
VX   = 200 / 3.6
PHIT = 0.1

[FX, FY, FZ, MX, MY, MZ, SL, SA, IA, PHI, VX, P, RE, rho, a, t, mu_x, mu_y, N, RL, b, MZR, Cx, Cy, Cz, KYA, sigma_x,
 sigma_y, iKYA, KXK] = tyre.find_full_output(SA=SA, SL=SL, FZ=FZ, VX=VX, P=P, IA=IA, PHIT=PHIT, angle_unit = "deg")

def rads2rpm(input):
    return input * 60.0 / (2.0 * np.pi)

print("=== FULL STATE OUTPUT ===")
print("Input state")
print(f"  Slip angle:           {SA:.1f} deg")
print(f"  Slip ratio:           {SL:.1f}")
print(f"  Inclination angle:    {IA:.1f} deg")
print(f"  Tyre pressure:        {1e-5*P:.2f} bar")
print(f"  Turn slip:            {PHI:.1f} /m")

print("Speed")
print(f"  Longitudinal:         {3.6*VX:.1f} km/h")
print(f"  Angular:              {rads2rpm(N):.1f} rpm")

print("Forces")
print(f"  Longitudinal:         {FX:.1f} N")
print(f"  Lateral:              {FY:.1f} N")
print(f"  Vertical:             {FZ:.1f} N")

print("Moments")
print(f"  Overturning:          {MX:.1f} Nm")
print(f"  Rolling resistance:   {MY:.1f} Nm")
print(f"  Self-aligning:        {MZ:.1f} Nm")
print(f"  Residual MZ:          {MZR:.1f} Nm")

print("Gradients")
print(f"  Cornering stiffness:  {np.deg2rad(KYA):.1f} N/deg")
print(f"  Slip stiffness:       {1e-2*KXK:.1f} N/0.01slip")

print(f"Friction coefficients")
print(f"  Longitudinal:         {mu_x:.3f}")
print(f"  Lateral:              {mu_y:.3f}")

print("Relaxation lengths")
print(f"  Longitudinal:         {1e3*sigma_x:.1f} mm")
print(f"  Lateral:              {-1e3*sigma_y:.1f} mm")

print("Radii and deflection")
#print(f"  Free radius:          {1e3*R_omega:.1f} mm")
print(f"  Loaded radius:        {1e3*RL:.1f} mm")
print(f"  Effective radius:     {1e3*RE:.1f} mm")
print(f"  Vertical deflection:  {1e3*rho:.1f} mm")

print("Stiffness")
print(f"  Longitudinal:         {1e-3*Cx:.1f} N/mm")
print(f"  Lateral:              {1e-3*Cy:.1f} N/mm")
print(f"  Vertical:             {1e-3*Cz:.1f} N/mm")

print("Contact patch dimensions")
print(f"  Length:               {1e3*a:.1f} mm")
print(f"  Width:                {1e3*b:.1f} mm")
print(f"  Pneumatic trail:      {1e3*t:.1f} mm")
