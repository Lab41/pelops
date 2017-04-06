from keras.models import Model
from keras.layers import Dense, Input, merge
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
        self.pca.fit(train_exp_gen.dataset.feats)
        kwargs['feature_transformer'] = self.pca.transform

        super().__init__(train_exp_gen,
                 test_exp_gen,
                 num_experiments,
                 *args,
                 **kwargs)


    def define_model(self):
        processed_left = Input(shape=[self.output_size])
        processed_right = Input(shape=[self.output_size])

        siamese_join = merge([processed_left, processed_right], mode='concat')
        my_layer = Dense(self.output_size, activation='relu')(siamese_join)
        my_layer = BatchNormalization()(my_layer)

        my_layer = Dense(self.output_size/2, activation='relu')(my_layer)

        num_training_classes=3
        predictions = Dense(num_training_classes, activation='sigmoid')(my_layer)

        self.model = Model([processed_left, processed_right], output=predictions)

        self.model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
