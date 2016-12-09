""" turn the list of files into a list of json dicts about the files

Input:
    Take the VeRi datset that contains the following information:
    * 49358 images (1679 query images, 11580 test images, 37779 train images)
    * 776 vehicles
    * 20 cameras
    * covering 1.0 km^2 area in 24 hours

    convert the name_* files into json files for processing

Output:
    json file with the following attributes in a dict per line:
        imageName
        vehicleID
        cameraID
        colorID
        typeID

Usage:
    veriFileList2Json [-hv]
    veriFileList2Json  -i <INFILE_NAME>

Arguments:
    INFILE_NAME         :file path to the VeRI name_ file

Options:
    -h, --help          :Show this message
    -v, --version       :Version of the prog
    -i, --inputFile     :location of the VeRi name_ file to process



"""
import docopt
import json
import sys


# turn the list of files into json for working with
def main(args):
    try:
        inFileName = args['--inputFile']
    except docopt.DocoptExit as e:
        sys.exit('error: input invalid options: {0}'.format(e))

    outFileName = '{0}.json'.format(inFileName)

    inFile = open(inFileName, 'r')
    outFile = open(outFileName, 'w')

    for line in inFile:
        d = dict()
        line = line.strip()
        attrs = line.split('_')
        d['imageName'] = line
        d['vehicleID'] = attrs[0]
        d['cameraID'] = attrs[1]
        d['colorID'] = str(-1)
        d['typeID'] = str(-1)
        outFile.write(json.dumps(d)+'\n')
    inFile.close()


if __name__ == '__main__':
    args = docopt.docopt(__doc__,version='veriFileList2Json 1.0')
    main(args)
