from abc import ABCMeta, abstractmethod
import numpy as np
from keras.models import model_from_json


class PelopsModel(metaclass=ABCMeta):
    """
    A base class for all Pelops Models
    """
    def __init__(self,
                 train_exp_gen,
                 test_exp_gen,
                 num_experiments,
                 *args,
                 **kwargs):
        """

        Args:
            train_exp_gen: Training data experiment generator
            test_exp_gen: Test data experiment generator
            num_experiments: Number of epxeriments to use for validation, training is 10x
            feature_transformer: A hook to transform feature vectors if desired
            truth_function: A function that takes two chips and returns the desired "truth"
                            e.g. same car or same car, color, vehicle type

        Returns:

        """
        self.train_exp_gen = train_exp_gen
        self.test_exp_gen = test_exp_gen
        self.num_experiments = num_experiments
        self.feature_transformer = kwargs.get('feature_transformer', lambda x: x)
        self.truth_function = kwargs.get('truth_function', PelopsModel.make_carid_type_color_truth)

    @abstractmethod
    def define_model(self):
        raise NotImplementedError()

    @staticmethod
    def make_carid_type_color_truth(chip1, chip2):
        """
        Takes two chips and returns if the chips represent the same [car_id, color, vehicle type]
        Args:
            chip1:
            chip2:

        Returns:
             if the two chips have the same [car_id, color, vehicle type]
        """
        same_vehicle = chip1.car_id == chip2.car_id
        same_type = chip1.misc['vehicle_type'] == chip2.misc['vehicle_type']
        same_color = chip1.misc['color'] == chip2.misc['color']
        return [same_vehicle, same_type, same_color]

    @staticmethod
    def make_carid_truth(chip1, chip2):
        """
        Takes two chips and returns if the chips represent the same car_id
        Args:
            chip1:
            chip2:

        Returns:
             if the two chips have the same [car_id]
        """
        same_vehicle = chip1.car_id == chip2.car_id
        return [same_vehicle]

    @staticmethod
    def make_batch(experiment_generator,
                    batch_size,
                    feature_transformer,
                    truth_function):
        """
        Make a set of training or test data to be used

        Args:
            experiment_generator: Pelops.experiment_api.experiment.Experiment
            batch_size: Number of examples to create
            feature_transformer: A hook to transform feature vectors if desired
            truth_maker: A function that takes two chips and returns the desired "truth"
                            e.g. same car or same car, color, vehicle type

        Returns:
            [Input
        """
        truths = []
        left_feats = []
        right_feats = []

        for i in range(batch_size):
            # Generate Example
            cam0, cam1 = experiment_generator.generate()

            # Find true match
            true_match = set([x.car_id for x in cam0]) & set([x.car_id for x in cam1])

            # Figure out which car in camera 0 is the "true" match
            for car in cam0:
                if car.car_id in true_match:
                    true_car = car

            true_car_feats = experiment_generator.dataset.get_feats_for_chip(true_car)

            # Construct examples
            for car_num, right_car in enumerate(cam1):
                truth = truth_function(true_car, right_car)
                right_car_feat = experiment_generator.dataset.get_feats_for_chip(right_car)

                # Add forward example
                left_feats.append(true_car_feats)
                right_feats.append(right_car_feat)
                truths.append(truth)

                # Add reversed example
                left_feats.append(right_car_feat)
                right_feats.append(true_car_feats)
                truths.append(truth)

        left_feats = feature_transformer(np.array(left_feats))
        right_feats = feature_transformer(np.array(right_feats))
        return [left_feats, right_feats], np.array(truths, dtype=np.uint8)

    def prep_train(self):
        self.X_train, self.Y_train = self.make_batch(self.train_exp_gen,
                                                     10*self.num_experiments,
                                                     self.feature_transformer,
                                                     self.truth_function)

    def prep_test(self):
        self.X_test, self.Y_test = self.make_batch(self.test_exp_gen,
                                                   self.num_experiments,
                                                   self.feature_transformer,
                                                   self.truth_function)

    def prep(self):
        self.define_model()
        self.prep_train()
        self.prep_test()

    def train(self,
              epochs,
              batch_size=128,
              callbacks=None):
        self.model.fit(self.X_train,
                       self.Y_train,
                       validation_data=(self.X_test, self.Y_test),
                       batch_size=batch_size,
                       nb_epoch=epochs,
                       callbacks=callbacks,
                       verbose=2)

    def save(self, base_filename):
        json_filename = base_filename + '.json'
        weights_filename = base_filename + '.weights'

        # serialize model to JSON
        model_json = self.model.to_json()
        with open(json_filename, 'w') as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save_weights(weights_filename)

    def load(self, base_filename):
        json_filename = base_filename + '.json'
        weights_filename = base_filename + '.weights'

        # load json and create model
        json_file = open(json_filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model .load_weights(weights_filename)
