import sys
from pelops.datasets.veri import VeriDataset
from pelops.etl.extract_feats_from_chips import extract_feats_from_chips

# make the stuff that we run on
if __name__ == '__main__':
    # path to the veri dataset
    v_file_name = sys.argv[0]

    # filename of where to place the output
    out_file_name = sys.argv[1]

    veri = VeriDataset(v_file_name)
    extract_feats_from_chips(veri, out_file_name)
