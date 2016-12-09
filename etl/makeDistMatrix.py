import json
import sys
from multiprocessing import Pool
import scipy.spatial.distance
import itertools
import numpy as np
import time


# read the list of things to compare
def makeWork(vectorFileName):
    vfile = open(vectorFileName, 'r')
    retval = list()
    for line in vfile:
        line = line.strip()
        line = json.loads(line)
        retval.append(line)
    vfile.close()
    return retval


# help by chopping work into chunks
def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


# my distance measures
def my_dist(workList):
    retval = list()

    for pair in workList:
        x = pair[0]
        y = pair[1]
        fx = np.asarray(x['resnet50'])
        fy = np.asarray(y['resnet50'])
        workItem = dict()
        dc = str(float(scipy.spatial.distance.cosine(fx, fy)))
        de = str(float(scipy.spatial.distance.euclidean(fx, fy)))
        workItem['x'] = x['imageName']
        workItem['y'] = y['imageName']
        workItem['cosine'] = dc
        workItem['euclidean'] = de
        retval.append(workItem)

    return (retval)


# takes in a json file with vectors and creates all the pairwise
# distance calculations, saves output to file
def main(pworkers=15, atOnceOuter=100000, atOnceInner=10000):

    inFileName = sys.argv[1]
    work = makeWork(inFileName)
    p = Pool(pworkers)

    outFileName = 'matrixFile.{0}'.format(inFileName)
    matrixFile = open(outFileName, 'w')

    total = 0
    for batch in grouper(atOnceOuter, itertools.combinations(work, 2)):
        start = time.time()
        batched = list()

        for workbatch in grouper(atOnceInner, batch):
            batched.append(workbatch)

        retval = p.map(my_dist, batched)
        end = time.time()
        start2 = time.time()
        for listLine in retval:
            for line in listLine:
                total = total + 1
                matrixFile.write(json.dumps(line)+'\n')
        end2 = time.time()

    fstr = 'proc elapsed:{0} sec proc:{1} total{2}'
    print(fstr.format(end-start, atOnceOuter, total))
    print('IO elapsed:{0}\n'.format(end2-start2))
    matrixFile.close()

if __name__ == '__main__':
    main()
