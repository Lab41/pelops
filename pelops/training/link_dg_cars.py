from pelops.training.utils import KerasDirectory, key_make_model, key_make_model_color
from pelops.datasets.dgcars import DGCarsDataset
from pelops.utils import SetType


DATASET_PATH = "/path/to/dgcar/dataset/"
OUTPUT_PATH = "/path/to/output/make_model_color/"

TYPES = (
    SetType.TRAIN,
    SetType.TEST,
    SetType.ALL,
)

for settype in TYPES:
    cc = DGCarsDataset(DATASET_PATH, settype)
    kd = KerasDirectory(cc, key_make_model_color)
    kd.write_links(OUTPUT_PATH)
