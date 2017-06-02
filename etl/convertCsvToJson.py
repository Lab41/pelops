"""
Conversion script for image2vecs feature vector csvs to siamese json

Environment Variables:
- pelops_csv_*: one or more file paths to csvs for conversion
- pelops_csv_mode:
  - 'product': Combine using the cartesian product of the records from 2x csvs [default]
  - 'combo': Combine using pair-wise combinations of records for each csv (1 or more)
- pelops_json: Path to output json file
"""

import os
import sys
import traceback
from pelops.utils import prep_for_siamese

if __name__ == '__main__':
    csv_files = [v for k, v in os.environ.items() if k.startswith('pelops_csv') and os.path.isfile(v)]

    if len(csv_files) == 0:
        print("No CSV files were provided for conversion")
        sys.exit(-1)
    print("Converting {} csv files:\n\t - {}".format(len(csv_files), '\n\t - '.join(csv_files)))

    mode = os.getenv('pelops_csv_mode', 'product')
    print("Mode: {}".format(mode))

    out_json = os.getenv('pelops_json', None)
    if out_json is None:
        print("Output json file path was not specified")
    print("Json: {}".format(out_json))

    try:
        prep_for_siamese(*csv_files, json_file=out_json, full_combos=(mode != 'product'))
        print("Conversion success")
    except:
        print("Conversion error occurred:\n{}".format(traceback.format_exc()))
