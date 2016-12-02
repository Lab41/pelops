""" Transform images in a directory

Usage:
    transform.py [-hv]
    transform.py [-a -b -g -s PERCENTAGE] <INPUT_PATH> <OUTPUT_PATH>

Arguments:
    INPUT_PATH                      : Path to the input directory where images are stored
    OUTPUT_PATH                     : Path to the output directory where resulting, degraded images are saved

Options: 
    -h, --help                      : Show the help message
    -v, --version                   : Show the version number
    -a, --all                       : Perform all the degraded functions to the images
    -b, --blur                      : Blur the images
    -s PERCENTAGE, --ste=PERCENTAGE : Shrink by PERCENTAGE (default is 25) and then enlarge
    -g, --grayscale                 : Convert the images into grayscale

"""

import docopt
import os
import sys
import time

from PIL import Image, ImageFilter
from resizeimage import resizeimage

def main(args={}):
    """ Transform images in a directory
    Args: 
        args: docopt arguments passed by the user via command line; there are 3 required arguments
        1. input_path: path to input directory where images are stored
        2. output_path: path to output directory where degraded images are saved 
        3. operation options to degrade the images
    """

    # extract arguments from command line
    is_blur = False
    is_ste = False
    ste_percentage = 25
    is_grayscale = False
    try:
        input_path = args["<INPUT_PATH>"]
        output_path = args["<OUTPUT_PATH>"]
        if args["--all"]:
            is_blur = True
            is_ste = True
            is_grayscale = True
        if args["--blur"]:
            is_blur = True
        if args["--ste"]:
            is_ste = True
            ste_percentage = float(args["--ste"])
        if args["--grayscale"]:
            is_grayscale = True
    except docopt.DocoptExit as e:
        sys.exit("ERROR: input invalid options: %s" % e)

    # check that input_path points to a directory
    if not os.path.exists(input_path):
        sys.exit("ERROR: input path (%s) is invalid" % input_path)
    if not os.path.isdir(input_path):
        sys.exit("ERROR: require input path (%s) to be a directory" % input_path)

    # create the output directory if it did not already exist
    makedir(output_path)

    # create the subdirectories within the output directory 
    # for each degradation we will performs
    if is_blur:
        blur_dir = makepath(os.path.abspath(output_path), "blur"); 
        makedir(blur_dir)
    if is_ste:
        ste_dir = makepath(os.path.abspath(output_path), "shrink_then_enlarge"); 
        makedir(ste_dir)
    if is_grayscale:
        grayscale_dir = makepath(os.path.abspath(output_path), "grayscale"); 
        makedir(grayscale_dir)

# -----------------------------------------------------------------------------
#  Transformation functions
# -----------------------------------------------------------------------------

    # define the functions that will transform the image
    class TransformedImage(object):
        def __init__(self, img, output_path):
            self.img = img

        def blur(self):
            blur_img = self.img.filter(ImageFilter.BLUR)
            blur_img_path = makepath(blur_dir, os.path.basename(self.img.filename))
            blur_img.save(blur_img_path)
            return

        def shrink_then_enlarge(self, percentage=25):
            width, height = self.img.size
            resized_width = int((percentage/100.0) * width)
            resized_height = int((percentage/100.0) * height)

            # shrink by 50% with degradation
            shrink_img = self.img.thumbnail((resized_width, resized_height))
            # without degradation: self.img.thumbnail((5,5), Image.ANTIALIAS)
            
            # enlarge to original size
            ste_img = self.img.resize((width , height))
            ste_img_path = makepath(ste_dir, os.path.basename(self.img.filename))
            ste_img.save(ste_img_path)
            # without degradation: ste_img.save(ste_img_path, quality=90)
            return

        def grayscale(self):
            grayscale_img = self.img.convert("1")
            grayscale_img_path = makepath(grayscale_dir, os.path.basename(self.img.filename))
            grayscale_img.save(grayscale_img_path)
            return 

# -----------------------------------------------------------------------------
#  Execution
# -----------------------------------------------------------------------------

    # carry out the degradation
    # not wrapping it in try-catch to get better error messages
    for file_path in traverse(input_path):
        img = TransformedImage(Image.open(file_path), output_path)
        print("----------------------------------------------------------------")
        print("Image: %s" % img.img.filename)
        if is_blur: print("Complete blur in %0.3f seconds" % timeit(img.blur))
        if is_grayscale: print("Complete grayscale in %0.3f seconds" % timeit(img.grayscale))
        if is_ste: print("Complete shrink then enlarge in %0.3f seconds" % timeit(img.shrink_then_enlarge, ste_percentage))
    return

# -----------------------------------------------------------------------------
#  Helper functions
# -----------------------------------------------------------------------------

def timeit(func, *args, **kwargs):
    """ This is a wrapper function to calculate how fast each operation takes.
    Note that we are assuming that we do not need to do anything with the return
    value of the function.
    Args: 
        func: function pointer
        args: arguments to the function
        kwargs: named arguments not defined in advance to be passed in to the function
    """
    start = time.time()
    func(*args, **kwargs)
    elapsed = time.time() - start
    return elapsed

def makepath(dirname, basename):
    """ Create a full path given the directory name and the base name
    Args:
        dirname: path to directory
        basename: name of file or directory
    """
    return dirname + "/" + basename

def makedir(path):
    """ Create a directory if it did not already exist
    Args: 
        path: path where the directory will be created
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return

def traverse(directory):
    """ Use os.scandir to traverse the directory faster 
    Args: 
        directory: path to a directory
    Returns:
        generator that lists the path to all the files in the directory
    """
    for entry in os.scandir(directory):
        if entry.is_file():
            yield entry.path

# -----------------------------------------------------------------------------
#  Entry
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    args = docopt.docopt(__doc__, version="Transform Images 1.0")
    main(args)

