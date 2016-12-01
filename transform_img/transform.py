""" Transform images in a directory

Usage:
    transform.py 
    transform.py (-h | --help)

Arguments:
    INPUT_PATH      Path to the input directory where images are stored
    OUTPUT_PATH     Path to the output directory where resulting, degraded images are saved

Options: 
    --all           Perform all the degraded functions to the images
    --blur          Blur the images
    --ste           Shrink by 25% (default) and then enlarge
    --grayscale     Convert the images into grayscale

"""

import docopt
import os
import sys

from PIL import Image, ImageFilter
from resizeimage import resizeimage

def main(args):
    """ Transform images in a directory
    Args: 
        input_path: path to input directory where images are stored
        output_path: path to output directory where degraded images are saved 
    """

    # extract 
    try:
        input_path = args['INPUT_PATH']
        output_path = args['OUTPUT_PATH']
    except docopt.DocoptExit as e:
        sys.exit("ERROR: input invalid options: %s" % e)


    # check that input_path points to a directory
    if not os.path.exists(input_path):
        sys.exit("ERROR: input path (%s) is invalid" % input_path)
    if not os.path.isdir(input_path):
        sys.exit("ERROR: require input path (%s) to be a directory" % input_path)

    # create the output directory if it did not already exist
    makedir(output_path)


    input_path = "/Users/tiffanyj/datasets/pelops/cars/cars/images"
    output_path = "/Users/tiffanyj/datasets/pelops/degraded/cars"
    main(input_path, output_path)


    # create the subdirectories within the output directory 
    # for each degradation we will perform
    blur_dir = makepath(os.path.abspath(output_path), "blur"); makedir(blur_dir)
    ste_dir = makepath(os.path.abspath(output_path), "shrink_then_enlarge"); makedir(ste_dir)
    grayscale_dir = makepath(os.path.abspath(output_path), "grayscale"); makedir(grayscale_dir)

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
    for file_path in traverse(input_path):
        try:
            img = TransformedImage(Image.open(file_path), output_path)
            img.blur()
            img.shrink_then_enlarge()
            img.grayscale()
        except IOError:
            print ("ERROR: file (%s) is not an image" % file_path)
        except Exception as e:
            sys.exit("ERROR: %s" % e.message)
    return

# -----------------------------------------------------------------------------
#  Helper functions
# -----------------------------------------------------------------------------

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
    main(docopt.docopt(__doc__))

