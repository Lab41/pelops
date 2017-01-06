import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cosine, euclidean


# compute cosine distance
# 0 -> things are closer
# 1 -> things are far
def comp_cosine(cam1_feat, cam2_feat):
    retval = 1 - cosine(cam1_feat, cam2_feat)
    return (retval)


# compute euclidian distance
# 0 -> things are closer
# + -> thins are far
def comp_euclid(cam1_feat, cam2_feat):
    retval = abs(euclidean(cam1_feat, cam2_feat))
    return (retval)


# do the comparisons between chips
# cam1 - listing of chips seen at cam1
# cam2 - listing of chips seen at cam1
# comparison - function to compare 2 vectors should return small things
#              when comparison is close, large otherwise
# verbose - return more info if true
def is_correct_match(featureData,
                     cam1,
                     cam2,
                     comparison=comp_cosine, verbose=False):
    similarities = []
    for cam1_chip in cam1:
        cam1_feat = featureData.get_feats_for_chip(cam1_chip)
        for cam2_chip in cam2:
            cam2_feat = featureData.get_feats_for_chip(cam2_chip)
            similarity = comparison(cam1_feat, cam2_feat)
            similarities.append((similarity, cam1_chip, cam2_chip))
    similarities.sort(reverse=True)
    for i, (similarity, chip1, chip2) in enumerate(similarities):
        # return best_match
        if chip1.car_id == chip2.car_id:
            if verbose:
                return i, similarities
            else:
                return i
    raise ValueError("Huh?")


# do EXPPERCMC, determine
# featureData - big table to look up data
# experimentGen  - function to create experiments
# EXPPERCMC - number of experiments to run for a single CMC
# comparison - function to compare 2 feature vectors
def pre_cmc(featureData, experimentGen,
            EXPPERCMC=1000, comparison=comp_cosine):

    num_downs = defaultdict(int)
    for i in range(EXPPERCMC):
        a = experimentGen.generate()
        num_down = is_correct_match(featureData, a[0], a[1],
                                    comparison=comparison)
        num_downs[num_down] += 1

    keys = sorted(num_downs)
    vals = [num_downs[key] for key in keys]
    return((keys, np.array(vals)/EXPPERCMC))


# Generate unprocessed CMC curves
# the data needs to be summed to make the correct
# CMC curve
# featureData - FeatureDataset of chips
# experimentGen - ExperimentGenerator
# NUMCMC - number of CMC to build
# EXPPERCMC - number of experiments run per CMC
# comparison - function that compares two feature vectors returning
#              distance measure, 0 -> close  big -> far
def repeat_pre_cmc(featureData, experimentGen, NUMCMC=100,
                   EXPPERCMC=1000, comparison=comp_cosine):
    experimentHolder = []
    for experiment in range(NUMCMC):
        experimentHolder.append(pre_cmc(featureData, experimentGen,
                                        EXPPERCMC=EXPPERCMC,
                                        comparison=comparison))
    return experimentHolder


# finalize creation of the CMC curves
# generate statistics on the CMC curves
# return all
# experimentHolder - array of CMC curves
# itemsPerCamera - number of items on a camera
def make_cmc_stats(experimentHolder, itemsPerCamera):
    comparisons = itemsPerCamera*itemsPerCamera
    stats = np.zeros((len(experimentHolder), comparisons))

    for index, (keys, vals) in enumerate(experimentHolder):
        for keyIndex in range(len(keys)):
            stats[index, keys[keyIndex]] = vals[keyIndex]

    for index in range(len(stats[:, ])):
        total_sum = 0.0
        offsetlen = len(stats[0])
        for sample in range(offsetlen):
            total_sum += stats[index, sample]
            stats[index, sample] = total_sum

    gdata = np.zeros((3, comparisons))

    for i in range(comparisons):
        gdata[1, i] = np.average(stats[:, i])
    for i in range(comparisons):
        stddev = np.std(stats[:, i])
        gdata[0, i] = gdata[1, i] - stddev
        gdata[2, i] = gdata[1, i] + stddev

    return (stats, gdata)
