import keras
from keras import layers
from keras import models
from keras import optimizers
import datetime

class Network(object):
    """docstring for Network."""
    def __init__(self, k_features=64, n_items=10000, n_users=1000, n_train_samples=10, n_test_samples=10, optimizer='sgd', loss='categorical_crossentropy'):
        super(Network, self).__init__()
        self.k_features = k_features
        self.n_items = n_items
        self.n_users = n_users
        self.n_classes = 5
        self.n_train_samples = n_train_samples
        self.n_test_samples  = n_test_samples
        self.optimizer = optimizer
        self.loss = loss
        self.model = self.model_func()
        self.model_type = "Basic"
        self.descr = self.description_str()

    def model_func(self):
        raise NotImplementedError()

    def description_str(self, suffix="", uid=False):
        model_type = self.model_type + "_"
        model_size = str(self.n_train_samples) + "_train_" + str(self.n_test_samples) + "_test_" + str(self.k_features) + "_features_"
        model_params = self.optimizer + "_" + self.loss + "_"
        model_categ = ""
        if self.n_classes > 1:
            model_categ = "categorical_"
        model_uid = ""
        if uid is True:
            model_uid = datetime.datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
        return model_type + model_size + model_params + model_categ + suffix

class ShallowNetwork(Network):
    """docstring for ShallowNetwork."""
    def __init__(self, *args, **kwargs):
        super(ShallowNetwork, self).__init__(*args, **kwargs)
        self.model_type = "Shallow"

    def model_func(self):
        input_i = layers.Input(shape=[1])
        i = layers.Embedding(self.n_items+1, self.k_features)(input_i)
        i = layers.Flatten()(i)
        i = layers.normalization.BatchNormalization()(i)

        input_u = layers.Input(shape=[1])
        u = layers.Embedding(self.n_users+1, self.k_features)(input_u)
        u = layers.Flatten()(u)
        u = layers.normalization.BatchNormalization()(u)

        nn = layers.concatenate([i, u])

        nn = layers.Dense(512, activation='relu')(nn)
        nn = layers.Dropout(0.5)(nn)
        nn = layers.normalization.BatchNormalization()(nn)

        nn = layers.Dense(128, activation='relu')(nn)

        output = layers.Dense(self.n_classes, activation='softmax')(nn)

        model = models.Model([input_i, input_u], output)
        model.compile(optimizer=self.optimizer, loss=self.loss)
        return model

class DeepNetwork(Network):
    """docstring for ShallowNetwork."""
    def __init__(self, *args, **kwargs):
        super(DeepNetwork, self).__init__(*args, **kwargs)
        self.model_type = "Simple_Deep"

    def model_func(self):
        input_i = layers.Input(shape=[1])
        i = layers.Embedding(self.n_items + 1, self.k_features)(input_i)
        i = layers.Flatten()(i)
        i = layers.normalization.BatchNormalization()(i)

        input_u = layers.Input(shape=[1])
        u = layers.Embedding(self.n_users + 1, self.k_features)(input_u)
        u = layers.Flatten()(u)
        u = layers.normalization.BatchNormalization()(u)

        nn = layers.concatenate([i, u])

        nn = layers.Dense(1024, activation='relu')(nn)
        nn = layers.Dropout(0.5)(nn)
        nn = layers.normalization.BatchNormalization()(nn)
        nn = layers.Dense(512, activation='relu')(nn)
        nn = layers.Dropout(0.5)(nn)
        nn = layers.normalization.BatchNormalization()(nn)
        nn = layers.Dense(256, activation='relu')(nn)
        nn = layers.Dropout(0.5)(nn)
        nn = layers.normalization.BatchNormalization()(nn)
        nn = layers.Dense(128, activation='relu')(nn)

        output = layers.Dense(5, activation='softmax')(nn)

        model = models.Model([input_i, input_u], output)
        model.compile(optimizer=self.optimizer, loss=self.loss)
        return model
