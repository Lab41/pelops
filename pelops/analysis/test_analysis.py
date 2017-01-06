import pytest
from pelops.analysis import analysis


class experimentGen():
    def __init__(self):
        self.fd = featureData()
        self.experiment = list()
        c1 = ['a', 'b', 'c', 'd']
        c2 = ['a', 'e', 'f', 'g']
        cam1 = list()
        cam2 = list()

        for c in c1:
            cam1.append(self.fd.getchip(c))

        for c in c2:
            cam2.append(self.fd.getchip(c))

        self.experiment.append(cam1)
        self.experiment.append(cam2)

    def generate(self):
        return self.experiment


class chip():
    def __init__(self, x):
        self.car_id = x[0]
        self.feature = x[1]


class featureData():
    def __init__(self):
        self.data = list()

        fun = [('a', [1, 2, 3, 4, 5, 6, 7]),
               ('b', [10, 20, 30, 40, 11, 9, 2.7]),
               ('c', [100, 20, 30, 40, 11, 9, 2.7]),
               ('d', [10, 200, 30, 40, 11, 9, 2.7]),
               ('e', [10, 20, 300, 40, 11, 9, 2.7]),
               ('f', [10, 20, 30, 400, 11, 9, 2.7]),
               ('g', [10, 20, 30, 40, 110, 9, 2.7]),
               ('h', [10, 20, 30, 40, 11, 90, 2.7]),
               ('i', [10, 20, 30, 40, 11, 9, 27.0])]
        for f in fun:
            self.data.append(chip(f))

    def get_feats_for_chip(self, chip):
        for d in self.data:
            if d.car_id == chip.car_id:
                return d.feature

    def getchip(self, id):
        for d in self.data:
            if d.car_id == id:
                return d

# test the comparisons


def test_cosine():
    a = [1, 2, 3, 4, 5, 6, 7]
    b = [10, 20, 30, 40, 11, 9, 2.7]
    out = analysis.comp_cosine(a, b)
    assert(abs(out - 0.63837193721375185) < 0.0000001)


def test_euclidian():
    a = [1, 2, 3, 4, 5, 6, 7]
    b = [10, 20, 30, 40, 11, 9, 2.7]
    out = analysis.comp_euclid(a, b)
    assert(abs(out - 49.93485756463114) < 0.0000001)

# test the matching works correctly


def test_is_correct_match():
    fd = featureData()

    c1 = ['a', 'b', 'c', 'd']
    c2 = ['a', 'e', 'f', 'g']
    cam1 = list()
    cam2 = list()

    for c in c1:
        cam1.append(fd.getchip(c))

    for c in c2:
        cam2.append(fd.getchip(c))

    out = analysis.is_correct_match(fd, cam1, cam2)
    assert (out == 0)


def test_pre_cmc():
    eg = experimentGen()
    fd = featureData()
    keys, values = analysis.pre_cmc(fd, eg, EXPPERCMC=10)
    assert values[0] == 1.0


#test the statistics are being generated correctly
def test_make_cmc_stats():
    eg = experimentGen()
    fd = featureData()
    experimentHolder = analysis.repeat_pre_cmc(fd, eg, NUMCMC=10, EXPPERCMC=10)
    stats, gdata = analysis.make_cmc_stats(experimentHolder, 4)

    for x in range(len(gdata[0])):
        assert ( gdata[1][x] ==gdata[2][x] == gdata[0][x])
