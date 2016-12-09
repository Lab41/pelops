import json
import sys

# turn the list of files into json for working with
def main():
    inFileName = sys.argv[1]
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
    main()
