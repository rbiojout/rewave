"""
Train a classifier given optimal action
"""

from __future__ import absolute_import
from __future__ import print_function

import os

from datetime import date

import numpy as np

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input, Dense, Flatten
from tensorflow.python.keras.models import Model

from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import to_categorical

from rewave.learn.root_network import RootNetwork

from rewave.environment.long_portfolio import PortfolioEnv
from rewave.tools.data import create_optimal_imitation_dataset, create_imitation_dataset
from rewave.tools.paths import get_root_model_path, get_root_model_dir_path, get_model_path, get_result_path


def create_network_given_future(state_input,
                                predictor_type="cnn", use_batch_norm=False,
                                weight_path='weights/root_model/optimal_stocks.h5',
                                load_weights=False):

    root_network = RootNetwork(inputs=state_input, predictor_type=predictor_type, use_batch_norm=use_batch_norm).net
    root_model = Model(state_input, root_network)

    if load_weights:
        try:
            root_model.load_weights(weight_path)
            for layer in root_model.layers:
                layer.trainable = False
            print('Model load successfully')
        except:
            print('Build model from scratch')
    else:
        print('Build model from scratch')

    net = Dense(300, activation="relu")(root_network)
    net = Flatten()(net)
    net = Dense(300, activation="relu")(net)

    nb_assets = state_input.shape[1]
    output = Dense(nb_assets, activation='softmax')(net)

    imitation_model = Model(state_input, output)

    imitation_model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    return root_model, imitation_model


def train_optimal_action_given_future_obs(model, target_history, target_stocks,
                                          weight_path='weights/optimal_3_stocks.h5'):
    (X_train, y_train), (X_test, y_test) = create_optimal_imitation_dataset(target_history)
    nb_classes = len(target_stocks) + 1

    Y_train = to_categorical(y_train, nb_classes)
    Y_test = to_categorical(y_test, nb_classes)

    continue_train = True
    while continue_train:
        model.fit(X_train, Y_train, batch_size=128, epochs=50, validation_data=(X_test, Y_test), shuffle=True)
        save_weights = input('Type True to save weights\n')
        if save_weights:
            model.save(weight_path)
        continue_train = input('True to continue train, otherwise stop\n')


def create_network_give_past(nb_classes, window_length, weight_path='weights/imitation_3_stocks.h5'):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(1, 3), input_shape=(nb_classes, window_length, 1),
                     activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=32, kernel_size=(1, window_length - 2), input_shape=(nb_classes, window_length - 2, 1),
                     activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten(input_shape=(window_length, nb_classes)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    try:
        model.load_weights(weight_path)
        print('Model load successfully')
    except:
        print('Build model from scratch')
    return model


def train_optimal_action_given_history_obs(model, target_history, target_stocks, window_length,
                                           weight_path='weights/imitation_3_stocks.h5'):
    nb_classes = len(target_stocks) + 1
    (X_train, y_train), (X_validation, y_validation) = create_imitation_dataset(target_history, window_length)
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_validation = np_utils.to_categorical(y_validation, nb_classes)
    X_train = np.expand_dims(X_train, axis=-1)
    X_validation = np.expand_dims(X_validation, axis=-1)
    continue_train = True
    while continue_train:
        model.fit(X_train, Y_train, batch_size=128, epochs=100, validation_data=(X_validation, Y_validation),
                  shuffle=True)
        save_weights = input('Type True to save weights\n')
        if save_weights:
            model.save(weight_path)
        continue_train = input("True to continue train, otherwise stop training...\n")


tickers_list = ['AAPL', 'ATVI', 'CMCSA', 'COST', 'CSX', 'DISH', 'EA', 'EBAY', 'FB', 'GOOGL', 'HAS', 'ILMN', 'INTC', 'MAR', 'REGN', 'SBUX']
features_list = ['open', 'high', 'low', 'close']

window_length = 15
predictor_type="cnn"
use_batch_norm = False


def test_main():
    print("STARTING MAIN")
    start_date = date(2008, 1, 1)
    end_date = date(2017, 1, 1)

    # setup environment
    num_training_time = 2000
    env = PortfolioEnv(start_date, end_date,
                       window_length,
                       tickers_list, features_list, batch_size=num_training_time)

    nb_assets = len(tickers_list) + 1
    nb_features = len(features_list)
    action_dim = env.action_space.shape[0]

    root_model_dir_path = get_root_model_dir_path(window_length, predictor_type, use_batch_norm)
    root_model_path = get_root_model_path(window_length, predictor_type, use_batch_norm)

    # create the network
    state_input = Input(shape=(nb_assets, window_length, nb_features), name="state_input")
    root_model, imitation_model = create_network_given_future(state_input,
                                predictor_type="cnn", use_batch_norm=False,
                                weight_path=root_model_path, load_weights=True)

    history = env.src.full_history()
    # create the dataset
    (X_train, y_train), (X_validation, y_validation) = create_optimal_imitation_dataset(history, training_data_ratio=0.8)

    # train
    history = imitation_model.fit(X_train, y_train, batch_size=128, epochs=100, validation_data=(X_validation, y_validation), shuffle=True)
    os.makedirs(root_model_dir_path, exist_ok=True)
    # remove last layer before saving
    root_model.save_weights(root_model_path)

if __name__ == '__main__':
    test_main()
