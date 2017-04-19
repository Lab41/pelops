""" Generate resnet50 features

Input:
    infile shold be a list of json lines one json/line

Output:
    appending of resnet50 features to each json line

Usage:
    makeFeaturesResNet50 [-hv]
    makeFeaturesResNet50 -i <INPUT_FILENAME> -p <IMAGE_DIR>

Arguments:
    INPUT_FILENAME        : location of the file to enrich with resnet features
    IMAGE_DIR             : full path to where the images live

Options:
    -h, --help            : Show this help message.
    -v, --version         : Show the version number.
    -i, --inFile          : input file to enrich with reset fetures
    -p, --path            : Path to the directory holding the images


"""

import docopt
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
import os
import time
import json
import sys


# return an image from a file, default resize to 224,224
def load_image(img_path, resizex=224, resizey=224):
    data = image.load_img(img_path, target_size=(resizex, resizey))
    x = image.img_to_array(data)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


# load the imagenet networks
def get_models():
    # include_top needs to be True for this to work
    base_model = ResNet50(weights='imagenet', include_top=True)
    model = Model(input=base_model.input,
                  output=base_model.get_layer('flatten_1').output)
    return (model, base_model)


# return feature vector for a given img, and model
def image_features(img, model):
    features = model.predict(img)
    return features


# read the files to process
def getList(name):
    retval = list()
    f = open('name', 'r')
    for line in f:
        line = line.strip()
        line = json.loads(line)
        retval.append(line)
    f.close()
    return retval


# perform the file by file processing
def process(trainingList, prefix, model, outFilename, batchSize=1000):
    outFile = open(outFilename, 'w')
    start = time.time()
    for idx, line in enumerate(trainingList):
        tempd = dict()
        if idx % batchSize == 0:
            end = time.time() - start
            start = time.time()
            fstring = 'total {0} batch {1} images in {2} seconds'
            print (fstring.format(idx, batchSize, end))
            path = os.path.join(prefix, line['imageName'])
            img = load_image(path)
            feature = image_features(img, model)
            tempd['resnet50'] = feature.tolist()[0]
            tempd.update(line)
            outFile.write(json.dumps(tempd)+'\n')
    outFile.close()


# read json file append feature vector to each line dict
def main(args):
    try:
        lineFileName = args['--inFile']
        prefix = args['--path']

    except docopt.DocoptExit as e:
        sys.exit('Error: input invalid options {0}'.format(e))

    outFilename = '{0}.resnet50.json'.format(lineFileName)
    model, base_model = get_models()

    print('loading...')
    trainingList = getList(lineFileName)

    print('processing...')
    process(trainingList, prefix,  model, outFilename)

    print('done.')


if __name__ == '__main__':
    args = docopt.docopt(__doc__, version='1.0')
    main(args)
