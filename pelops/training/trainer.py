from random import choice
from sklearn.linear_model import LogisticRegression as LR
import pickle

from pelops.experiment_api.experiment import ExperimentGenerator


class ModelTrainer(object):
    """Train an sklearn model using a FeatureDataset() dataset. """

    def __init__(
            self,
            model,
            vector_combiner,
            train_ds,
            test_ds=None,
            n_test=10000,
            n_train=1000,
            random_seed=1024,
    ):
        """Set up a class for training sklearn classification models.

        Agrs:
            model: An instantiated sklearn classification model that provides
                `.fit(X, Y)` and `.score(X, Y)`,
            vector_combiner: A function that takes two feature vectors and
                returns an iterable of vectors.
            train_ds: a FeatureDataset() of data to use for training.
            test_ds (defaults to None): a FeatureDataset() of data to use for
                testing. May be set to None as it is only used if .score() is
                called.
            n_train (int, defaults to 10,000): Number of training datapoint to construct.
            n_test (int, defaults to 1000): Number of testing datapoint to construct.
            random_seed (int, defaults to 1024): Seed for the experiment
                generators.

        """
        # Set up Datasets and experiment generators
        self.train_ds = train_ds
        self.train_eg = None
        if self.train_ds is not None:
            self.train_eg = ExperimentGenerator(self.train_ds, num_cams=2, num_cars_per_cam=2, drop_percentage=0, seed=random_seed)

        self.test_ds = test_ds
        self.test_eg = None
        if self.test_ds is not None:
            self.test_eg = ExperimentGenerator(self.test_ds, num_cams=2, num_cars_per_cam=2, drop_percentage=0, seed=random_seed)

        self.model = model
        self.join = vector_combiner
        self.trained_model = None

        self.n_test = n_test
        self.n_train = n_train

        # Make the datasets
        self.X_train, self.Y_train = self.__generate_datasets(self.train_ds, self.train_eg, self.n_train)
        # We only initialize the test if we want it
        self.X_test, self.Y_test = None, None

    def __generate_datasets(self, dataset, generator, n):
        X = []
        Y = []

        while len(X) < n:
            camset = generator.generate()
            voi0 = dataset.get_feats_for_chip(camset[0][0])
            voi1 = dataset.get_feats_for_chip(camset[1][0])
            bg0  = dataset.get_feats_for_chip(camset[0][1])
            bg1  = dataset.get_feats_for_chip(camset[1][1])

            # Append a True match
            x_poses = self.join(voi0, voi1)
            for x in x_poses:
                X.append(x)
                Y.append(True)

            # Append a False match
            #
            # Randomly select the background car, but always use the 0th VOI as
            # this gives the same result as randomly selecting both.
            bg = choice((bg0, bg1))
            x_negs = self.join(voi0, bg)
            for x in x_negs:
                X.append(x)
                Y.append(False)

        return X, Y

    def fit(self):
        """Fit the model to the training data. Sets self.trained_model."""
        self.trained_model = self.model.fit(self.X_train, self.Y_train)

    def score(self):
        """Return the score of a trained model.

        You must first call .fit() or this call will fail. test_ds must be a
        valid FeatureDataset() as well.
        """
        if self.trained_model is not None:
            # If the test data has not be generated, generate it
            if self.X_test is None or self.Y_test is None:
                if self.test_eg is None:
                    raise TypeError("test_ds must be an FeatureDataset to run .score()")
                self.X_test, self.Y_test = self.__generate_datasets(self.test_ds, self.test_eg, self.n_test)

            # Test the model using the test data
            return self.trained_model.score(self.X_test, self.Y_test)

        raise RuntimeWarning("model is not trained, try calling .fit()")

    def save(self, file_name):
        """Save a model to a text file using pickle."""
        if self.trained_model is not None:
            with open(file_name, 'wb') as f:
                pickle.dump(self.trained_model, f)
        else:
            print("Can not save untrained model! Call .fit() first.")
