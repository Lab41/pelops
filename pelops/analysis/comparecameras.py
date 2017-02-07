""" camera comparison """

import itertools
from collections import defaultdict

import numpy as np
from tqdm import tnrange

from pelops.analysis.camerautil import (get_match_id, glue, make_good_bad,
                                        nameit_cam, nameit_car)


def eval_good_bad(first, second, clf, featuredataset, goodmatches, badmatches, attribute_name):
    """
    label examples of good and bad comparisons

    take two chips, concantenate their feature vectors
    and create a balanced dataset of matches and differences

    first(Chip):  image to evaluate
    second(Chip): image to evaluate
    clr(classifier): classifier used to evaluate chips
    fd(featureDataset): maps chips to features
    goodmatches(defaultdictionary(int)): counts of good matches
    badmatches(defaultdictionary(int)): counts of bad matches
    attribute_name(str): which attribute to pull names from
    """

    namefunc = None
    if attribute_name == 'car':
        namefunc = nameit_car
    else:
        namefunc = nameit_cam

    bigvec1 = glue(featuredataset.get_feats_for_chip(first),
                   featuredataset.get_feats_for_chip(second))

    bigvec1np = np.array(bigvec1)
    #bigvec1np.reshape(1, -1)

    bigvec2 = glue(featuredataset.get_feats_for_chip(second),
                   featuredataset.get_feats_for_chip(first))

    bigvec2np = np.array(bigvec2)
    # bigvec2np.reshape(1, -1))

    decision = clf.predict(bigvec1np.reshape(1, -1))
    name = namefunc(first, second)

    tally_decision(decision, goodmatches, name, badmatches)

    decision = clf.predict(bigvec2np.reshape(1, -1))
    name = namefunc(second, first)

    tally_decision(decision, goodmatches, name, badmatches)


def tally_decision(decision, goodpic, name, badpic):
    """
    count the number of matches for a name

    decision(int): whether the classifier said they matched
    goodpic(defaultdict(int)): list of good matches
    badpic(defaultdict(int)): list of bad matches
    name(str): concatenation of names of first and second pics
    """
    if decision == 1:
        goodpic[name] += 1
    else:
        badpic[name] += 1


def mad_matrix(examples, clf, featuredataset, examplegenerator, attribute_name='car'):
    """
    run examples experiments to see how cars are declaired
    the same or different by the clf classifier.abs

    examples(int): number of trials
    clf(classifier): classifier to make same/different distinciton
    fd(featureDataset) : allows joining of chip to features
    eg(experimentGenerator): makes expermients for testing
    """

    ddg = defaultdict(int)
    ddb = defaultdict(int)

    for _ in tnrange(examples):
        cameras_test = examplegenerator.generate()
        match_id = get_match_id(cameras_test)
        goods, bads = make_good_bad(cameras_test, match_id)
        good0 = goods[0]
        good1 = goods[1]
        bad0 = bads[0]
        bad1 = bads[1]

        eval_good_bad(good0, good1, clf, featuredataset,
                      ddg, ddb, attribute_name)
        eval_good_bad(bad0, bad1, clf, featuredataset,
                      ddb, ddg, attribute_name)

    return(ddg, ddb)


def make_work(fd_train, lessons, outcomes, items, label):
    """
    makes a listing of work from chips for classification

    fd_train(featureDataset): training features
    lessons(list): feature vectors
    outcomes(list): expected outcome for the comparison
    items(list(chips)): list of chips for comparison
    label(int): expected label for the comparison
    """
    workitems = itertools.permutations(items, 2)
    for workitem in workitems:
        item = glue(fd_train.get_feats_for_chip(
            workitem[0]), fd_train.get_feats_for_chip(workitem[1]))

        lessons.append(item)
        outcomes.append(label)

        item = glue(fd_train.get_feats_for_chip(
            workitem[1]), fd_train.get_feats_for_chip(workitem[0]))

        lessons.append(item)
        outcomes.append(label)
