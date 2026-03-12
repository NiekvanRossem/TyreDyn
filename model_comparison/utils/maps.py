# list containing the output signals of MFeval
ORDER_MFEVAL = ["FX", "FY", "FZ", "MX", "MY", "MZ", "SL", "SA", "IA", "PHIT", "VX", "P", "RE", "rho", "a", "t", "mu_x",
                "mu_y", "N", "RL", "b", "MZR", "Cx", "Cy", "Cz", "KYA", "sigma_x", "sigma_y", "iKYA", "KXK"]

# list containing the output signals of TyreDyn
ORDER_TYREDYN = ["FX", "FY", "FZ", "MX", "MY", "MZ", "SL", "SA", "IA", "PHIT", "VX", "P", "N", "R_omega", "RE", "rho",
                 "RL", "a", "b", "t", "mu_x", "mu_y", "MZR", "Cx", "Cy", "Cz", "KYA", "iKYA", "KXK", "iKXK", "sigma_x",
                 "sigma_y"]

# list of parameters in display order
PARAMETERS = [
    "SA", "SL", "IA", "P", "PHIT", "VX", "N",
    "FX", "FY", "FZ", "MX", "MY", "MZ", "MZR",
    "KYA", "KXK", "mu_x", "mu_y", "sigma_x", "sigma_y",
    "RL", "RE", "rho", "Cx", "Cy", "Cz", "a", "b", "t"
]

# map with channel names
NAMES_MAP = {
    "SA" : "slip angle", "SL" : "slip ratio", "IA" : "inclination angle", "P" : "pressure", "PHIT" : "turn slip",
    "VX" : "longitudinal speed", "N" : "angular speed", "FX" : "longitudinal force", "FY" : "lateral force",
    "FZ" : "vertical force", "MX" : "overturning moment", "MY" : "rolling resistance moment",
    "MZ" : "self-aligning moment", "MZR" : "residual aligning moment", "KYA" : "cornering stiffness",
    "KXK" : "slip stiffness", "mu_x" : "longitudinal friction", "mu_y" : "lateral friction",
    "sigma_x" : "longitudinal relaxation", "sigma_y" : "lateral relaxation", "RL" : "loaded radius",
    "RE" : "effective radius", "rho" : "vertical deflection", "Cx" : "longitudinal stiffness",
    "Cy" : "lateral stiffness", "Cz" : "vertical stiffness", "a" : "contact patch length", "b" : "contact patch width",
    "t" : "pneumatic trail"
}

# map with display units per channel
UNITS_MAP = {
    "SA": "deg   ", "SL": "-     ", "IA": "deg   ", "P": "bar   ", "PHIT": "/m    ", "VX": "km/h  ", "N": "rpm   ",
    "FX": "N     ", "FY": "N     ", "FZ": "N     ", "MX": "Nm    ", "MY": "Nm    ", "MZ": "Nm    ", "MZR": "Nm    ",
    "KYA": "N/deg ", "KXK": "e2 N/slip", "mu_x": "-     ", "mu_y": "-     ", "sigma_x": "mm    ", "sigma_y": "mm    ",
    "RL": "mm    ", "RE": "mm    ", "rho": "mm    ", "Cx": "N/mm  ", "Cy": "N/mm  ", "Cz": "N/mm  ", "a": "mm    ",
    "b": "mm    ", "t": "mm    "
}

# sections for the display table
SECTIONS = [
    "Input state",
    "Speed",
    "Forces",
    "Moments",
    "Gradients",
    "Friction coefficients",
    "Relaxation lengths",
    "Radii and deflection",
    "Stiffness",
    "Contact patch dimensions"
]