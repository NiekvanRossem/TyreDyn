from tyredyn.types.model_map import MODEL_CLASS_MAP
from tyredyn.infrastructure.paths import TYRE_DIR
from tyredyn.io.normalize_fittyp import normalize_fittyp
from tyredyn import *
from tyredyn.io.tir_validation import TIRValidation
from pathlib import Path
from typing import Union

class Tyre:
    """
    Initialization class for tyre models. This class contains the functions to read and validate a TIR file. Not
    to be confused with ``TyreBase``, which contains the template for the subclasses.

    Currently supported tyre models:
      - MF-Tyre 6.1
      - MF-Tyre 6.2

    Settings
    --------
    ``check_format`` : bool, optional
        Checks the shape of the input arrays, and flattens them if needed (default is ``True``).
    ``check_limits`` : bool, optional
        Checks if all values in the input array are within the specified limits (default is ``True``).
    ``print_status`` : bool, optional
        Set to `False` if you don't want to print status messages after loading (default is `True`).
    ``validate`` : bool, optional
        Perform validation on TIR file parameters (default is ``True``).
    ``use_alpha_star`` : bool, optional
        Slip angle correction for large angles and reverse running (default is ``True``).
    ``use_gamma_star`` : bool, optional
        Inclination angle correction for large angles (default is ``True``).
    ``use_turn_slip`` : bool, optional
        Turn slip correction (default is ``False``).
    ``use_lmu_star`` : bool, optional
        Composite friction scaling factor with slip speed (default is ``True``).

    Parameters
    ----------
    filename : string
        Path to the TIR file.

    Returns
    -------
    tyre : class
        Instance of a tyre class with the desired model type.
    """

    def __new__(cls, filepath: str, **settings):

        # option to disable printing messages after loading in
        print_status: bool = settings.get('print_status', True)

        # overwrite FITTYP if the user selected it
        use_model_type: str = settings.get('use_model_type', None)

        # option to disable validation procedure
        validate: bool = settings.get('validate', True)

        # read TIR file
        params = cls.read_tir(filepath)
        if use_model_type is not None:
            model_type = normalize_fittyp(use_model_type)
        else:
            model_type = normalize_fittyp(params.get("MODEL", {}).get("FITTYP", None))
        if print_status:
            print(f"TIR file '{filepath}' successfully loaded.")
        if validate:
            cls._validate_data(params)

        # map the FITTYP parameter to the right subclass
        try:
            subclass = MODEL_CLASS_MAP[model_type]
        except KeyError:
            raise KeyError(f"Model type '{model_type}' not recognized. Cannot create the desired instance.")

        # create instance of the new subclass
        obj = super().__new__(subclass)
        obj.__init_from_data__(params, **settings)
        subclass.__init__(obj, params, **settings)
        if print_status:
            print(f"Tyre instance of type {subclass} successfully created.")
        return obj

    def __init__(self, filepath, **settings):
        pass

    @staticmethod # TODO: move
    def read_tir(filepath: Union[str, Path]) -> dict:
        """
        Reads the TIR file, and store parameters as a dictionary.

        Template written by ChatGPT, and modified from there.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path of the TIR file to be read.

        Returns
        -------
        params : dict
            Dictionary of parameter names and values.
        """

        # change string to path if needed
        if isinstance(filepath, str):
            filepath = Path(filepath)

        # extract filename
        filename = filepath.name

        # if the directory does not exist, try the example tyres folder
        if not filepath.is_file():
            filepath = TYRE_DIR / filename

        # raise an error if it still does not exist
        if not filepath.is_file():
            raise FileNotFoundError(f"File '{filepath}' not found.")

        # read file
        with open(TYRE_DIR / filepath) as f:
            data = f.readlines()
            params = {}
            params_list = []

            # loop over all the lines
            for line in data:
                line = line.strip()

                # non-tyres_example line
                if line.startswith('$-') or line.startswith('!'):
                    continue

                # start of a new section (create dict entry for it)
                if line.startswith('[') and line.endswith(']'):
                    current_header = line[1:-1]
                    params[current_header] = {}

                # line contains useful tyres_example
                elif '=' in line:

                    # filter out the comment if there is one
                    if '$' in line and not line.startswith('$'):
                        line, _ = line.split('$', 1)

                    # add to dictionary
                    key, value = line.split('=')
                    params[current_header][key.strip()] = value.strip()
                    params_list.append(key.strip())

            # convert all the numbers to floats, except FITTYP
            for header in params.keys():
                for key in params[header].keys():
                    try:
                        if key == "FITTYP":
                            continue
                        else:
                            params[header][key] = float(params[header][key])

                    except ValueError:
                        continue

        params.pop("MDI_HEADER", None)
        return params

    @staticmethod # TODO: move
    def _validate_data(params):
        """
        Validate the TIR file.

        :param params: Dictionary of parameter names and values.
        """
        TIRValidation.validate_with_model(params)

#----------------------------------------------------------------------------------------------------------------------#

# test script
if __name__ == "__main__":

    from tyredyn import Tyre
    import numpy as np

    filepath = Path(r'/tyres_example\MF62\car205_60R19.tir')

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
    SA = 3.0
    SL = 0.05
    FZ = 4500.0
    P = 1.8e5
    IA = 0.0
    VX = 100 / 3.6
    PHIT = 0.1
    #SA = 0.0
    #SL = 0.0
    #FZ = -100.0
    #P = 1.8e5
    #IA = 0.0
    #VX = 200 / 3.6
    #PHIT = 0.0

    [FX, FY, FZ,
     MX, MY, MZ,
     SL, SA, IA, PHIT, VX, P, N,
     R_omega, RE, rho, RL,
     a, b, t,
     mu_x, mu_y,
     MZR,
     Cx, Cy, Cz,
     KYA, iKYA, KXK, iKXK,
     sigma_x, sigma_y] = tyre.find_full_output(SA=SA, SL=SL, FZ=FZ, VX=VX, P=P, IA=IA, PHIT=PHIT, angle_unit="deg")

    # planar force
    FH = np.sqrt(FX ** 2 + FY ** 2)
    mu_ix = np.abs(FX / (FZ + 1e-12))
    mu_iy = np.abs(FY / (FZ + 1e-12))
    mu_i  = np.abs(FH / (FZ + 1e-12))

    def radps2rpm(input):
        return input * 60.0 / (2.0 * np.pi)

    print("=== FULL STATE OUTPUT ===")
    print("Input state")
    print(f"  Slip angle:           {SA:.1f} deg")
    print(f"  Slip ratio:           {SL:.2f}")
    print(f"  Inclination angle:    {IA:.1f} deg")
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
    print(f"  Longitudinal (inst):  {mu_ix:.3f}")
    print(f"  Longitudinal:         {mu_x:.3f}")
    print(f"  Lateral (inst):       {mu_iy:.3f}")
    print(f"  Lateral:              {mu_y:.3f}")
    print(f"  Planar (inst):        {mu_i:.3f}")

    print("Relaxation lengths")
    print(f"  Longitudinal:         {1e3 * sigma_x:.3f} mm")
    print(f"  Lateral:              {-1e3 * sigma_y:.3f} mm")

    print("Radii and deflection")
    print(f"  Free radius:          {1e3 * R_omega:.3f} mm")
    print(f"  Loaded radius:        {1e3 * RL:.3f} mm")
    print(f"  Effective radius:     {1e3 * RE:.3f} mm")
    print(f"  Vertical deflection:  {1e3 * rho:.3f} mm")

    print("Stiffness")
    print(f"  Longitudinal:         {1e-3 * Cx:.3f} N/mm")
    print(f"  Lateral:              {1e-3 * Cy:.3f} N/mm")
    print(f"  Vertical:             {1e-3 * Cz:.3f} N/mm")

    print("Contact patch dimensions")
    print(f"  Length:               {1e3 * a:.3f} mm")
    print(f"  Width:                {1e3 * b:.3f} mm")
    print(f"  Pneumatic trail:      {1e3 * t:.3f} mm")
