% Prepare workspace
clear; clc; close all;

% Load tyre
filepath = "C:\Users\niekv\Documents\5. Personal engineering scripts\TyreDyn\tyres_example\car205_60R19.tir";
tyre = mfeval.readTIR(filepath);

% input state
SA = 3.0 * pi / 180;
SL = 0.05;
FZ = 4500.0;
P = 1.8e5;
IA = 0.0;
VX = 200 / 3.6;
PHIT = 0.0;

% evaluate tyre
out = mfeval(tyre, [FZ, SL, SA, IA, PHIT, VX, P], 122);

% extract output data
FX      = out(1);
FY      = out(2);
FZ      = out(3);
MX      = out(4);
MY      = out(5);
MZ      = out(6);
SL      = out(7);
SA      = out(8);
IA      = out(9);
PHIT    = out(10);
VX      = out(11);
P       = out(12);
RE      = out(13);
rho     = out(14);
a       = out(15);
t       = out(16);
mu_x    = out(17);
mu_y    = out(18);
N       = out(19);
RL      = out(20);
b       = out(21);
MZR     = out(22);
Cx      = out(23);
Cy      = out(24);
Cz      = out(25);
KYA     = out(26);
sigma_x = out(27);
sigma_y = out(28);
iKYA    = out(29);
KXK     = out(30);

disp('=== FULL STATE OUTPUT ===')
disp('Input state')
disp(['  Slip angle:           ', num2str(rad2deg(SA)), ' deg'])
disp(['  Slip ratio:           ', num2str(SL)])
disp(['  Inclination angle:    ', num2str(rad2deg(IA)), ' deg'])
disp(['  Tyre pressure:        ', num2str(1e-5*P), ' bar'])
disp(['  Turn slip:            ', num2str(PHIT), ' /m'])

disp('Speed')
disp(['  Longitudinal:         ', num2str(3.6*VX), ' km/h'])
disp(['  Angular:              ', num2str(rads2rpm(N)), ' rpm'])

disp('Forces')
disp(['  Longitudinal:         ', num2str(FX), ' N'])
disp(['  Lateral:              ', num2str(FY), ' N'])
disp(['  Vertical:             ', num2str(FZ), ' N'])

disp('Moments')
disp(['  Overturning:          ', num2str(MX), ' Nm'])
disp(['  Rolling resistance:   ', num2str(MY), ' Nm'])
disp(['  Self-aligning:        ', num2str(MZ), ' Nm'])
disp(['  Residual MZ:          ', num2str(MZR), ' Nm'])

disp('Gradients')
disp(['  Cornering stiffness:  ', num2str(deg2rad(KYA)), ' N/deg'])
disp(['  Slip stiffness:       ', num2str(1e-2*KXK), ' N/0.01slip'])

disp('Friction coefficients')
disp(['  Longitudinal:         ', num2str(mu_x)])
disp(['  Lateral:              ', num2str(mu_y)])

disp('Relaxation lengths')
disp(['  Longitudinal:         ', num2str(1e3*sigma_x), ' mm'])
disp(['  Lateral:              ', num2str(1e3*sigma_y), ' mm'])

disp('Radii and deflection')
%disp(['  Free radius:          ', num2str(1e3*R_omega), ' mm'])
disp(['  Loaded radius:        ', num2str(1e3*RL), ' mm'])
disp(['  Effective radius:     ', num2str(1e3*RE), ' mm'])
disp(['  Vertical deflection:  ', num2str(1e3*rho), ' mm'])

disp('Stiffness')
disp(['  Longitudinal:         ', num2str(1e-3*Cx), ' N/mm'])
disp(['  Lateral:              ', num2str(1e-3*Cy), ' N/mm'])
disp(['  Vertical:             ', num2str(1e-3*Cz), ' N/mm'])

disp('Contact patch dimensions')
disp(['  Length:               ', num2str(1e3*a), ' mm'])
disp(['  Width:                ', num2str(1e3*b), ' mm'])
disp(['  Pneumatic trail:      ', num2str(1e3*t), ' mm'])

function output = rads2rpm(input)
    output = input * 60.0 / (2.0 * pi);
end