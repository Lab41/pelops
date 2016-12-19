""" This is an example use case.
"""

import chip
import veri

def print_out_dataset_types():
    dataset_types = [dataset.__name__ for dataset in chip.ChipDataset.__subclasses__()]
    print(dataset_types)
    # dataset_types = ["VeriDataset", "CompcarsDataset", "GoogleDataset", "StrDataset"]

def veri_example():
    dataset_type = "VeriDataset"
    dataset = chip.DatasetFactory.create_dataset(dataset_type, "")
    # check length
    """
    print("length: {}".format(len(dataset)))
    """
    # check iteration
    """
    for c in iter(dataset):
        print("name: {}".format(c.name))
        print("filepath: {}".format(c.filepath))
        print("car id: {}".format(c.car_id))
        print("camera id: {}".format(c.camera_id))
        print("timestamp: {}".format(c.get_timestamp()))
        print("binary: {}".format(c.binary))
        print("-" * 80)
    """
    # check get_all_chips_by_car_id()\
    """
    for c in dataset.get_all_chips_by_car_id(1):
        print("name: {}".format(c.name))
        print("filepath: {}".format(c.filepath))
        print("car id: {}".format(c.car_id))
        print("camera id: {}".format(c.camera_id))
        print("timestamp: {}".format(c.get_timestamp()))
        print("binary: {}".format(c.binary))
        print("-" * 80)
    """
    # check get_all_chips_by_camera_id()
    """
    for c in dataset.get_all_chips_by_camera_id(7):
        print("name: {}".format(c.name))
        print("filepath: {}".format(c.filepath))
        print("car id: {}".format(c.car_id))
        print("camera id: {}".format(c.camera_id))
        print("timestamp: {}".format(c.get_timestamp()))
        print("binary: {}".format(c.binary))
        print("-" * 80)
    """
    # get chip filepath by chip id
    
    return

def compcars_example():
    dataset_type = "CompcarsDataset"
    dataset = chip.DatasetFactory.create_dataset(dataset_type)
    return

def google_example():
    dataset_type = "GoogleDataset"
    dataset = chip.DatasetFactory.create_dataset(dataset_type)
    return

def strchip_example():
    dataset_type = "StrDataset"
    dataset = chip.DatasetFactory.create_dataset(dataset_type)
    return

if __name__ == "__main__":
    print_out_dataset_types()
    veri_example()
    #compcars_example()
    #google_example()
    #strchip_example()