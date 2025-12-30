from src.utils.model_map import MODEL_CLASS_MAP
from src.utils.paths import TYRE_DIR
from src.utils.formatting import *
from src.utils.tir_validation import TIRValidation

class Tyre:
    """
    Initialization class for tyre tyre_models. This class contains the functions to read and validate a TIR file. Not
    to be confused with ``TyreBase``, which contains the template for the subclasses.

    Currently supported tyre tyre_models:
      - MF 6.1
      - MF 6.2

    Parameters
    ----------
    filename : string
        Path to the TIR file.
    check_format : bool, optional
        Checks the shape of the input arrays, and flattens them if needed (default is ``True``).
    check_limits : bool, optional
        Checks if all values in the input array are within the specified limits (default is ``True``).
    print_status : bool, optional
        Set to `False` if you don't want to print status messages after loading (default is `True`).
    validate : bool, optional
        Perform validation on TIR file parameters (default is ``True``).
    use_alpha_star : bool, optional
        Slip angle correction for large angles and reverse running (default is ``True``).
    use_gamma_star : bool, optional
        Inclination angle correction for large angles (default is ``True``).
    use_turn_slip : bool, optional
        Turn slip correction (default is ``False``).
    use_lmu_star : bool, optional
        Composite friction scaling factor with slip speed (default is ``True``).

    Returns
    -------
    tyre : class
        Instance of a tyre class with the desired model type.
    """

    def __new__(cls, filename: str, **settings):

        # option to disable printing messages after loading in
        print_status: bool = settings.get('print_status', True)

        # overwrite FITTYP if the user selected it
        use_model_type: str = settings.get('use_model_type', None)

        # option to disable validation procedure
        validate: bool = settings.get('validate', True)

        # TODO: add support for custom paths
        # TODO: rename filename to filepath
        # TODO: figure out the best way to have the user select a default folder (make this part of initialize procedure)
        # 1. split filepath up into path and filename
        # 2. if only a filename is specified, try searching in the data folder
        # 3. if a filepath is specified, try to find the file there
        # 3. display an error message if file cannot be found

        # read TIR file
        params = cls._read_tir(filename)
        if use_model_type is not None:
            model_type = normalize_fittyp(use_model_type)
        else:
            model_type = normalize_fittyp(params.get("MODEL", {}).get("FITTYP", None))
        if print_status:
            print(f"TIR file '{filename}' successfully loaded.")
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

    def __init__(self, filename, **settings):
        pass

    @staticmethod
    def _read_tir(filename: str) -> dict:
        """
        Reads the TIR file, and store parameters as a dictionary.

        :param filename: Name of the TIR file to be read. Will assume this file is stored inside `data/tyres`.
        :return: Dictionary of parameter names and values.
        """

        # code for reading a TIR file. Outputs a dictionary with the tyre params.
        with open(TYRE_DIR / filename) as f:
            data = f.readlines()
            params = {}
            params_list = []

            # loop over all the lines
            for line in data:
                line = line.strip()

                # non-data line
                if line.startswith('$-') or line.startswith('!'):
                    continue

                # start of a new section (create dict entry for it)
                if line.startswith('[') and line.endswith(']'):
                    current_header = line[1:-1]
                    params[current_header] = {}

                # line contains useful data
                elif '=' in line:

                    # filter out the comment if there is one
                    if '$' in line and not line.startswith('$'):
                        line, _ = line.split('$', 1)

                    # add to dictionary
                    key, value = line.split('=')
                    params[current_header][key.strip()] = value.strip()
                    params_list.append(key.strip())

            # convert all the numbers to floats
            for header in params.keys():
                for key in params[header].keys():
                    try:
                        if key == "FITTYP":
                            continue
                        else:
                            params[header][key] = float(params[header][key])

                    except ValueError:
                        continue
        return params

    @staticmethod
    def _validate_data(params):
        """
        Validate the TIR file.

        :param params: Dictionary of parameter names and values.
        """
        TIRValidation.validate_with_model(params)

#----------------------------------------------------------------------------------------------------------------------#
# quick test script

if __name__ == "__main__":
    # initialize tyre
    tyre = Tyre(
        'car205_60R19.tir',
        validate        = True,
        use_model_type  = "MF62",
        use_alpha_star  = True,
        use_gamma_star  = True,
        use_lmu_star    = True,
        use_turn_slip   = False,
        check_format    = True,
        check_limits    = True,
        use_mfeval_mode = False
    )

    FZ = 600
    SL = 0.1

    FX = tyre.forces.find_fx_pure(SL=SL, FZ=FZ, VC=VC)

    #RL = tyre.find_loaded_radius(FX, 0.0, FZ, 20.0)
    print(f"=== TEST OUTPUT === \n"
          f"vertical load:  {FZ} N \n"
          f"slip ratio:     {SL} \n"
          f"tractive force: {FX:.2f} N \n")
          #f"loaded_radius:  {RL:.2f} m")