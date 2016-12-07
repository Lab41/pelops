"""
datasets contain 
* 50,000 images
* 776 vehicles
* 20 cameras
* 1.0 km^2 area
* 24 hours

vehicle_camera_

"""

import collections
import random
import re

def main(veri_unzipped_path):
    name_query = veri_unzipped_path + "name_query.txt"
    name_test = veri_unzipped_path + "name_test.txt"
    name_train = veri_unzipped_path + "name_train.txt"
    img_names = merge_names(name_query, name_test, name_train)
    print(get_num_spotted(img_names))
    num_cams = 5

    num_cars_per_cam = 3

    drop_percentage = 30

    seed = 11

    # -------------------------------------------------------------------------
    
    random.seed(seed)

    car_id = select_target_car(num_cams)
    camsets = collections.defaultdict()
    cam_ids = select_cameras(num_cams)


    
    return 



def should_drop(drop_percentage):
    """ Based on the given percentage, provide an answer 
    whether or not to drop the image.
    Args:
        drop_percentage: the likelihood of a drop 
    Returns:
        a boolean whether to drop or not drop the image
    """
    return random.randrange(100) < drop_percentage

def select_cameras(num_cams):
    """ Select camera ids based on the number of cameras specified.
    Args:
        num_cams: number of cameras 
    Returns:
        a list of camera ids
    """
    return random.sample(range(1, 21), num_cams)

# TODO: fix this to include camera!
def select_cars(num_cars, car_id_exception):
    """ Select car ids based on the number of cars specified. Do not
    include car_id_exception in the list.
    Args:
        num_cars: number of cars

    Returns:
        a list of car ids
    """
    valid_car_ids = list(range(1, 777))
    valid_car_ids.remove(car_id_exception)
    return random.sample(valid_car_ids, num_cars)

def select_target_car(num_cams):
    """ Select the target car for the experiment. The number of times the
    car is spotted by distinct cameras must be equal or greater than the 
    number of cameras (num_cams). num_cats create the number of 
    camera sets (camset) to return. If there are more camera sets than 
    there are the number of times the car is spotted, the drop percentage 
    is going to be higher. 
    Args:
        num_cams: 
    Returns:
        the car id of the target car
    """
    return random.randrange(1, 777)

def clean_num_spotted(num_cams):
    """ Only list cars who gets spotted at a specified num_cams or greater.
    Args:
    Returns:
        a counter that 
    """

# TODO: put this at initial state, you don't want to call this function multiple times
def get_num_spotted(img_names):
    """ Assuming that image name is in the format of carId_cameraId_timeId_binary.jpg, 
    count the number of times the car gets spotted by a distinct camera.
    Args:
        img_names: list of image names
    Returns:
        a counter that lists the number of times each car gets spotted by a camera
    """
    # extract a list of distinct cameras that spot the car
    cams_spotted = collections.defaultdict(set)
    for img_name in img_names:
        splitter = img_name.split('_')
        car_id = int(splitter[0])
        camera_id = int(get_numeric(splitter[1]))
        cams_spotted[car_id].add(camera_id)
    # -------------------------------------------------------------------------
    # TODO: if speed is crucial, remove the following and use len() during call 
    # to determine number of times distinct cameras spot the car
    # -------------------------------------------------------------------------
    # count the number of times distinct cameras spot the car
    num_spotted = collections.Counter()
    for k, v in cams_spotted.items():
        num_spotted[k] = len(v)
    return num_spotted

def get_numeric(string):
    """ Extract the numeric value in a string.
    Args:
        string
    Returns:
        a string with only the numeric value extracted
    """
    return re.sub('[^0-9]','', string)
     
def merge_names(*filepaths):
    """ Each file contains an image name per line. Merge all the image names
    in all the files so that we have a comprehensive list of image names.
    Since the data is not growing, store the names in a set for faster lookup.
    Args: 
        filepaths: paths to the files that contain an image name in each line
    Returns: 
        a set that contains the names of all the images
    """
    names = set()
    for filepath in filepaths:
        names = names.union(set(line.strip() for line in open(filepath)))
    return names

# -----------------------------------------------------------------------------
#  Check inputs
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
#  Entry
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    veri_unzipped_path = ""
    main(veri_unzipped_path)
