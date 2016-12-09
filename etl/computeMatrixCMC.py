import json
import time
from collections import defaultdict

from matplotlib import pyplot


def makeTransDicts(reindexFile):
    reindex = open(reindexFile, 'r')
    file2num = dict()
    num2file = dict()
    index = 0
    for line in reindex:
        line = line.strip()
        file2num[line] = index
        num2file[index] = line
        index += 1
    return (file2num, num2file)


def makeMatrix(matrixFilename, num2file, file2num, measure='cosine'):

    a = open(matrixFilename, 'r')
    lines = 0
    for line in a:
        lines += 1
    a.close()

    Matrix = [[0 for x in range(lines)] for y in range(lines)]
    matrixFile = open(matrixFilename, 'r')
    for line in matrixFile:

        line = line.strip()
        line = json.loads(line)
        x = file2num[line['x']]
        y = file2num[line['y']]
        Matrix[x][y] = line[measure]
        Matrix[y][x] = line[measure]

    for index in range(0, lines):
        Matrix[index][index] = 8675309
    return Matrix


def getrank(car, s, maxval=-1):
    for sidx, work in enumerate(s):
        # sval = work[0]
        scar = work[1]
        if scar == car:
            return sidx
    return maxval


def preCMC(Matrix, num2file, downto=50):
    retval = defaultdict(int)
    start = time.time()
    size = len(Matrix[0])

    for oindex in range(size):
        if oindex % 1000 == 0:
            print('index:{0} time:{1}'.format(oindex, time.time() - start))
            start = time.time()

        car = num2file[oindex].split('_')[0]

        current = list()

        for idx, val in enumerate(Matrix[oindex]):
            current.append((float(val), num2file[idx].split('_')[0]))

        s = sorted(current, key=lambda tup: tup[0])[:downto]
        maxSearch = downto + 1
        r = getrank(car, s, maxval=maxSearch)
        retval[r] += 1
    return retval


def computeCMC(rawCounts, num):
    idx = sorted(rawCounts)
    sum = 0
    CMC = list()
    for index in range(0, len(idx)):
        sum += rawCounts[index]
        print (index, sum)
        CMC.append(sum / float(num))
    return CMC


testFilesName = '/local_data/dgrossman/VeRi/test_uniqfiles'
matrixFilename = '/local_data/dgrossman/VeRi/matrixFile.test_uniqfile'
file2num, num2file = makeTransDicts(testFilesName)
Matrix = makeMatrix(matrixFilename, num2file, file2num)
rawCounts = preCMC(Matrix, num2file)
CMC = computeCMC(rawCounts, len(Matrix[0]))

# pyplot.ylim(0,1)
pyplot.plot(CMC[:-1])
pyplot.show()
