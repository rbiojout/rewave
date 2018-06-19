import tensorflow as tf

from keras import backend as K
from tensorflow.python.keras.models import Model

from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Permute, Activation, Flatten, Reshape, Lambda, LSTM
from tensorflow.python.keras.initializers import RandomUniform
from tensorflow.python.keras.activations import selu
from tensorflow.python.keras.losses import categorical_hinge


DEBUG = True

class RootNetwork(object):
    def __init__(self, inputs, predictor_type="cnn", use_batch_norm=False):
        self.inputs = inputs
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        assert inputs != None, 'Inputs must be defined'
        self.window_length = inputs.get_shape()[2]
        assert predictor_type in ['dense','cnn', 'lstm'], 'type must be either dense, cnn or lstm'
        self.net = self.build_network()

    def build_network(self):
        window_length = self.inputs.get_shape()[2]
        random_initializer = RandomUniform(minval=-0.05, maxval=0.05, seed=None)
        if self.predictor_type == 'dense':
            net = Dense(300, activation="relu")(self.inputs)
            # net = Dropout(0.3)(net)
            #net = BatchNormalization()(net)
            net = Dense(300, activation="relu")(net)
            # net = Dropout(0.3)(net)
            #net = BatchNormalization()(net)

            net = Flatten(name="root_network")(net)

        elif self.predictor_type == 'dense2':
            net = Dense(64, activation="relu", kernel_initializer=random_initializer)(self.inputs)
            if self.use_batch_norm:
                net = BatchNormalization()(net)

            net = Dense(64, activation="relu")(net)
            if self.use_batch_norm:
                net = BatchNormalization()(net)

            net = Flatten(name="root_network")(net)

        elif self.predictor_type == 'cnn':
            assert self.window_length >= 5, 'window length must be at least 3'
            net = Conv2D(filters=32, kernel_size=(1, 3),
                         padding='same',
                         data_format='channels_last',
                         activation='selu',
                         kernel_initializer = random_initializer)(self.inputs)
            if self.use_batch_norm:
                net = BatchNormalization()(net)
            #net = MaxPooling2D(pool_size=(2, 2))(net)
            #net = Dropout(0.25)(net)

            net = Conv2D(filters=32, kernel_size=(1, 3),
                         padding='valid',
                         data_format='channels_last',
                         activation='selu')(net)
            if self.use_batch_norm:
                net = BatchNormalization()(net)
            #net = MaxPooling2D(pool_size=(2, 2))(net)
            #net = Dropout(0.25)(net)

            if DEBUG:
                print('After conv2d:', net.shape)
            # net = Flatten(name="root_network")(net)
            if DEBUG:
                print('Output:', net.shape)
        elif self.predictor_type == 'lstm':
            num_stocks = self.inputs.get_shape()[1]
            window_length = self.inputs.get_shape()[2]
            features = self.inputs.get_shape()[3]

            hidden_dim = 32

            if DEBUG:
                print('Shape input:', self.inputs.shape)
            # net = Lambda(squeeze_middle2axes_operator, output_shape=squeeze_middle2axes_shape)(inputs)
            # net = Lambda(squeeze_first2axes_operator, output_shape=squeeze_first2axes_shape)(inputs)

            # (samples, time, filters, assets)
            net = Permute([2, 3, 1], name="permute_before_lstm_stage1")(self.inputs)
            if DEBUG:
                print('Shape input after transpose:', net.shape)

            # (samples, time, filters, output_row, output_col)
            net = Reshape((window_length, features * num_stocks), name="reshape_before_lstm_stage1")(net)


            # net = tf.squeeze(inputs, axis=0)
            # reorder
            # net = tf.transpose(net, [1, 0, 2])
            if DEBUG:
                print('Reshaped input:', net.shape)
            net = LSTM(units=hidden_dim, name="root_lstm_stage1")(net)
            if DEBUG:
                print('After LSTM:', net.shape)

            # net = Dropout(0.2, name="dropout_lstm_stage1")(net)
            if DEBUG:
                print('Output:', net.shape)
        else:
            raise NotImplementedError

        return net


def model_predictor(inputs, predictor_type, use_batch_norm):
    window_length = inputs.get_shape()[2]
    assert predictor_type in ['cnn', 'lstm'], 'type must be either cnn or lstm'
    assert window_length >= 3, 'window length must be at least 3'
