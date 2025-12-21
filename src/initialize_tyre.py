from src.utils.model_map import MODEL_CLASS_MAP
from src.utils.paths import TYRE_DIR
from src.utils.misc import *
from src.utils.tir_validation import TIRValidation

#from src.utils.read_file import read_tir
#from src.utils.tir_validation import validate_data

class Tyre:
    """
    Initialization class for tyre tyre_models. This class contains the functions to read and validate a TIR file. Not
    to be confused with ``TyreBase``, which contains the template for the subclasses.

    The following user-defined settings can be specified:
      - ``check_format`` -- checks the shape of the input arrays, and flattens them if needed (default is ``True``).
      - ``check_limits`` -- checks if all values in the input array are within the specified limits (default is ``True``).
      - ``print_status`` -- set to `False` if you don't want to print status messages after loading (default is `True`).
      - ``use_alpha_star`` -- slip angle correction for large angles and reverse running (default is ``True``).
      - ``use_gamma_star`` -- inclination angle correction for large angles (default is ``True``).
      - ``use_lmu_star`` -- composite friction scaling factor with slip speed (default is ``True``).

    Currently supported tyre models:
      - MF 6.1

    :param filename: filename of the TIR file
    :param validate: Set to True to validate the TIR file

    :return: An instance of the chosen tyre class, which can return the state of the tyre for a given input.
    """

    def __new__(cls, filename: str, validate: bool = True, **settings):

        # option to disable printing messages after loading in
        print_status: bool = settings.get('print_status', True)

        # TODO: add support for custom paths
        # TODO: rename filename to filepath
        # TODO: figure out the best way to have the user select a default folder (make this part of initialize procedure)
        # 1. split filepath up into path and filename
        # 2. if only a filename is specified, try searching in the data folder
        # 3. if a filepath is specified, try to find the file there
        # 3. display an error message if file cannot be found

        # read TIR file
        params = cls._read_tir(filename)
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
        if print_status:
            print(f"Tyre instance of type {subclass} successfully created.")
        return obj

    def __init__(self, filename, validate: bool = True, **settings):
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
    current_tyre = Tyre('car205_60R19.tir', validate=True, use_alpha_star=False, check_limits=False)

    FZ = 600
    SL = 0.1
    FX = current_tyre.find_fx_pure(SL, FZ, angle_unit='deg')
    RL = current_tyre.find_loaded_radius(FX, 0.0, FZ, 20.0)
    print(f"== TEST OUTPUT == \n"
          f"vertical load:  {FZ} N \n"
          f"slip ratio:     {SL} \n"
          f"tractive force: {FX:.2f} N \n"
          f"loaded_radius:  {RL:.2f} m")