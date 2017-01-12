from pelops.datasets.chip import ChipDataset
from pelops.utils import SetType
import os
import os.path



def attributes_to_classes(chip_dataset, chip_tuplizer):
    """Extract a set of attributes from a set of Chips and uses them to make
    unique classses.

    The chip_tuplizer is a function (or other callable) with the following
    signature:

        chip_tuplizer(chip) -> hashable tuple

    It returns a tuple derived from the chip. All chips that output the same
    tuple will be considered as part of the same class for training, where
    "sameness" is determined by hashing the tuple. An example tuplizer might do
    the following:

        chip_tuplizer(chip) -> (make, model)

    Args:
        chip_dataset: A ChipDataset, or other iterable of Chips
        chip_tuplizer: A function that takes a chip and returns a hashable
            tuple derived from the chip.

    Returns:
        dict: a dictionary mapping the output of chip_tuplizer(chip) to a
            class number.
    """
    class_to_index = {}
    current_index = 0
    for chip in chip_dataset:
        # Get the class from the specified attributes
        key = chip_tuplizer(chip)

        # If the key is new, add it to our dictionaries
        if key not in class_to_index:
            class_to_index[key] = current_index
            current_index += 1

    return class_to_index


def tuplize_make_model(chip):
    """ Given a chip, return make and model tuple.

    Make and model are extracted from chip.misc using the keys "make" and
    "model". If they are missing it returns None for that value. If misc
    missing or not a dictionary, (None, None) is returned.

    Args:
        chip: A chip named tuple

    Returns:
        tuple: (make, model) from the chip. None may be returned for one of the
            positions (or both) if it is missing in the chip.
    """
    # Ensure we have a misc dictionary
    try:
        misc = chip.misc
    except AttributeError:
        return (None, None)

    # Get the make and model
    try:
        make = misc.get("make", None)
        model = misc.get("model", None)
    except AttributeError:
        return (None, None)

    return (make, model)


def tuplize_color(chip):
    """ Given a chip, returns the color as a single element tuple.

    Color is extracted from chip.misc using the key "color". If it is missing
    or misc is not a dictionary, (None,) is returned.

    Args:
        chip: A chip named tuple

    Returns:
        tuple: (color,) from the chip. (None,) if not defined, or misc is
        missing.
    """
    # Ensure we have a misc dictionary
    try:
        misc = chip.misc
    except AttributeError:
        return (None,)

    # Get the make and model
    try:
        color = misc.get("color", None)
    except AttributeError:
        return (None,)

    return (color,)


def tuplize_make_model_color(chip):
    """ Given a chip, returns the make, model, and color tuple.

    Color is extracted from chip.misc using the keys "make, "model", and
    "color". If misc missing or not a dictionary, (None, None, None) is returned.

    Args:
        chip: A chip named tuple

    Returns:
        tuple: (make, model, color) from the chip. None may be returned for one
            of the positions (or any number of them) if it is missing in the
            chip.
    """
    (make_val, model_val) = tuplize_make_model(chip)
    (color_val,) = tuplize_color(chip)
    return (make_val, model_val, color_val)


class KerasDirectory(object):
    def __init__(self, chip_dataset, chip_tuplizer):
        """ Takes a ChipDataset and hard links the files to custom defined
        class directories.

        Args:
            chip_dataset: A ChipDataset, or other iterable of Chips
            chip_tuplizer: A callable that takes a chip and returns a tuple
                representing the attributes in that chip that you care about.
                For example, you might write a function to return the make and
                model, or maybe color.
        """
        # Set up internal variables
        self.__chip_dataset = chip_dataset
        self.__chip_tuplizer = chip_tuplizer

        # Class setup functions
        self.__set_root_dir()

        # Set up the class to index mapping
        self.__class_to_index = attributes_to_classes(
            self.__chip_dataset,
            self.__chip_tuplizer,
        )


    def __set_root_dir(self):
        """ Set the root directory for the classes based on the SetType.

        If self.__chip_dataset.set_type exists, it will be used to
        set the root directory name, otherwise it will default to
        "all".

        The final directory will be:

        output_directory / root / class_number / image
        """
        ROOTMAP = {
            SetType.ALL.value: "all",
            SetType.QUERY.value: "query",
            SetType.TEST.value: "test",
            SetType.TRAIN.value: "train",
        }

        # We write a train, test, query, or all directory as the root depending
        # on the ChipDataset.
        self.root = "all"
        try:
            set_type = self.__chip_dataset.set_type
        except AttributeError:
            return

        try:
            key = set_type.value
        except AttributeError:
            return

        try:
            self.root = ROOTMAP[set_type.value]
        except KeyError:
            return

    def write_links(self, output_directory, root=None):
        """ Writes links to a directory.

        The final directory will be:

        output_directory / root / class_number / image

        Where root is set by self.__set_root_dir() and is based on
        the SetType, but you can reset it by passing in root.

        Args:
            output_directory (str): The location to write the files to, it must
                already exist.
            root (str, Defaults to None): A base directory to create in the
                output_directory, under which all further directories will be
                written. If not specified, the class will choose between
                "test", "train", "query", and "all" depending on the `SetType`
                of the `chip_dataset`. If you would like no directory, use a
                blank string "".
        """
        # Override root with self.root if not set
        if root is None:
            root = self.root

        # Link chips
        for chip in self.__chip_dataset:
            src = chip.filepath
            filename = os.path.basename(src)
            chip_class = self.__chip_tuplizer(chip)
            chip_index = self.__class_to_index[chip_class]
            dest_dir = output_directory + "/" + root + "/" + str(chip_index) + "/"

            os.makedirs(dest_dir, exist_ok=True)
            os.link(src=src, dst=dest_dir + filename)
