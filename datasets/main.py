""" This is an example use case.
"""

import chip

def print_out_dataset_types():
    dataset_types = [dataset.__name__ for dataset in chip.ChipBase.__subclasses__()]
    print(dataset_types)
    # dataset_types = ["VeriDataset", "CompcarsDataset", "GoogleDataset", "StrDataset"]

def veri_example():
    dataset_type = "VeriDataset"
    dataset = chip.ChipFactory.create_dataset(dataset_type)
    # get all chips by car id
    # get all chips by camera id
    # get chip image path
    return

def compcars_example():
    dataset_type = "CompcarsDataset"
    dataset = chip.ChipFactory.create_dataset(dataset_type)
    return

def google_example():
    dataset_type = "GoogleDataset"
    dataset = chip.ChipFactory.create_dataset(dataset_type)
    return

def strchip_example():
    dataset_type = "StrDataset"
    dataset = chip.ChipFactory.create_dataset(dataset_type)
    return

if __name__ == "__main__":
    print_out_dataset_types()
    veri_example()
    compcars_example()
    google_example()
    strchip_example()