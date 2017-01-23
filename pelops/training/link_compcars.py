from pelops.training.utils import KerasDirectory, key_make_model
from pelops.datasets.compcar import CompcarDataset
from pelops.utils import SetType


DATASET_PATH = "/path/to/compcar/dataset/"
OUTPUT_PATH = "/path/to/output/make_model/"

TYPES = (
    SetType.TRAIN,
    SetType.TEST,
    SetType.ALL,
)

for settype in TYPES:
    cc = CompcarDataset(DATASET_PATH, settype)
    kd = KerasDirectory(cc, key_make_model)
    kd.write_links(OUTPUT_PATH)
