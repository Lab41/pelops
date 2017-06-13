import numpy as np

from keras.models import Model
from keras.layers import Dense, Input, merge, Reshape, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization

from sklearn import decomposition

from pelops.models.pelops_model import PelopsModel

class SiamesePCAModel(PelopsModel):
    def __init__(self,
                 train_exp_gen,
                 test_exp_gen,
                 num_experiments,
                 *args,
                 **kwargs):

        self.output_size = kwargs.get('pca_dim', 32)
        self.pca = decomposition.PCA(n_components=self.output_size)

        cars = set(train_exp_gen.list_of_cars)
        feats = []
        for chip in train_exp_gen.dataset.chips.values():
            if chip.car_id in cars:
                feats.append(train_exp_gen.dataset.get_feats_for_chip(chip))
        self.pca.fit(np.array(feats))
        kwargs['feature_transformer'] = self.pca.transform

        super().__init__(train_exp_gen,
                 test_exp_gen,
                 num_experiments,
                 *args,
                 **kwargs)


    def define_model(self):
        processed_left = Input(shape=[self.output_size])
        processed_right = Input(shape=[self.output_size])

        my_layer = merge([processed_left, processed_right], mode='concat')
        my_layer = Dense(self.output_size, activation='relu')(my_layer)
        my_layer = BatchNormalization()(my_layer)

        my_layer = Dense(self.output_size/2, activation='relu')(my_layer)

        num_training_classes=3
        predictions = Dense(num_training_classes, activation='sigmoid')(my_layer)

        self.model = Model([processed_left, processed_right], output=predictions)

        self.model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
