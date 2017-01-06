from pelops.datasets.chip import ChipDataset


def make_model(chip):
    """ Given a chip, return make and model.

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


def color(chip):
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


def make_model_color(chip):
    """ Given a chip, returns the make, model, and color.

    Color is extracted from chip.misc using the keys "make, "model", and
    "color". If misc missing or not a dictionary, (None, None, None) is returned.

    Args:
        chip: A chip named tuple

    Returns:
        tuple: (make, model, color) from the chip. None may be returned for one
            of the positions (or any number of them) if it is missing in the
            chip.
    """
    (make_val, model_val) = make_model(chip)
    (color_val,) = color(chip)
    return (make_val, model_val, color_val)
