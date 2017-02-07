""" work with SVM and chips """
import time

import sklearn
from scipy.stats import uniform as sp_rand
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tnrange

from pelops.analysis.camerautil import get_match_id, make_good_bad
from pelops.analysis.comparecameras import make_work


def train_svm(examples, fd_train, eg_train):
    """
    train a support vector machine

    examples(int): number of examples to generate
    fd_train(featureDataset): where to join features to chips
    eg_train(experimentGenerator): makes experiments

    clf(SVM): scm classifier trainined on the input examples
    """
    lessons_train = list()
    outcomes_train = list()
    for _ in tnrange(examples):
        cameras_train = eg_train.generate()
        match_id = get_match_id(cameras_train)
        goods, bads = make_good_bad(cameras_train, match_id)
        make_work(fd_train, lessons_train, outcomes_train, goods, 1)
        make_work(fd_train, lessons_train, outcomes_train, bads, 0)

    clf = svm.SVC()

    print('fitting')
    start = time.time()
    clf.fit(lessons_train, outcomes_train)
    end = time.time()
    print('fitting took {} seconds'.format(end - start))
    return clf


def search(examples, fd_train, eg_train, iterations):
    """
    beginnnings of hyperparameter search for svm
    """
    param_grid = {'C': sp_rand()}
    lessons_train = list()
    outcomes_train = list()
    for _ in tnrange(examples):
        cameras_train = eg_train.generate()
        match_id = get_match_id(cameras_train)
        goods, bads = make_good_bad(cameras_train, match_id)
        make_work(fd_train, lessons_train, outcomes_train, goods, 1)
        make_work(fd_train, lessons_train, outcomes_train, bads, 0)
    clf = svm.SVC()
    print('searching')
    start = time.time()
    rsearch = RandomizedSearchCV(
        estimator=clf, param_distributions=param_grid, n_iter=iterations)
    rsearch.fit(lessons_train, outcomes_train)
    end = time.time()
    print('searching took {} seconds'.format(end - start))
    print(rsearch.best_score_)
    print(rsearch.best_estimator_.C)


def save_model(model, filename):
    """
    save a model to disk

    model(somemodel): trained model to save
    filename(str): location to safe the model
    """
    joblib.dump(model, filename)


def load_model(filename):
    """
    load a model from disk. make sure that models only
    show up from version 0.18.1 of sklearn as other versions
    may not load correctly

    filename(str): name of file to load
    """
    if sklearn.__version__ == '0.18.1':
        model = joblib.load(filename)
        return model
    else:
        print('upgrade sklearn to version 0.18.1')


def test_svm(examples, clf_train, fd_test, eg_test):
    """
    score the trained SVM against test features

    examples(int): number of examples to run
    clf_train(modle): model for evaluating testing data
    fd_test(featureDataset): testing dataset
    eg_test(experimentGenerator): generated experiments from testing dataset

    out(int): score from the model
    """
    lessons_test = list()
    outcomes_test = list()

    for _ in tnrange(examples):
        cameras_test = eg_test.generate()
        match_id = get_match_id(cameras_test)
        goods, bads = make_good_bad(cameras_test, match_id)
        make_work(fd_test, lessons_test, outcomes_test, goods, 1)
        make_work(fd_test, lessons_test, outcomes_test, bads, 0)

    print('scoring')
    start = time.time()
    out = clf_train.score(lessons_test, outcomes_test)
    end = time.time()
    print('scoring took {} seconds'.format(end - start))
    return out
