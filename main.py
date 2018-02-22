from __future__ import absolute_import
import json
import logging
import os
import time
from argparse import ArgumentParser
from datetime import datetime
from datetime import date

import numpy as np

from rewave.marketdata.globaldatamatrix import HistoryManager
from rewave.marketdata.datamatrices import DataMatrices
from rewave.environment.long_portfolio import PortfolioEnv
from rewave.wrappers import SoftmaxActions, ConcatStates
from rewave.callbacks.tensorforce import EpisodeFinishedTQDM, EpisodeFinished
from rewave.tools.util import MDD, sharpe


from tensorforce.agents import Agent
from tensorforce.core.networks import LayeredNetwork, layers, Network, network
from tensorforce.core.networks import Layer, Conv2d, Nonlinearity
from tensorforce.core.baselines import NetworkBaseline
from tensorforce.core.baselines import baselines
from tensorforce.agents.ppo_agent import PPOAgent
from tensorforce.execution import Runner

import tensorflow as tf

start = date(2016, 1, 1)
end = date(2017, 1, 1)
start_test = date(2017, 1, 1)
end_test = date(2018, 1, 1)
window_size = 20
validation_portion = 0.15
feature_list = ['close', 'high', 'low', 'open']
feature_number = len(feature_list)
ticker_list = ['AAPL', 'A']
ticker_number = len(ticker_list)


from tensorforce.contrib.openai_gym import OpenAIGym
class TFOpenAIGymCust(OpenAIGym):
    def __init__(self, gym_id, gym):
        self.gym_id = gym_id
        self.gym = gym
        self.visualize = True


class EIIE(Layer):
    """
    EIIE layer
    """

    def __init__(self,
                 size=20,
                 bias=True,
                 activation='relu',
                 l2_regularization=0.0,
                 l1_regularization=0.0,
                 scope='EIIE',
                 summary_labels=()):
        self.size = size
        # Expectation is broadcast back over advantage values so output is of size 1
        self.conv1 = Conv2d(
            size=3,
            bias=bias,
            stride=(1, 1),
            window=(1, 3),
            padding='VALID',
            l2_regularization=l2_regularization,
            l1_regularization=l1_regularization,
            summary_labels=summary_labels)
        self.conv2 = Conv2d(
            size=size,
            bias=bias,
            stride=(1, window_size - 2 - 1),
            window=(1, window_size - 2 - 1),
            padding='VALID',
            l2_regularization=l2_regularization,
            l1_regularization=l1_regularization,
            summary_labels=summary_labels)
        self.conv3 = Conv2d(
            size=1,
            bias=bias,
            stride=(1, 1),
            window=(1, 1),
            l2_regularization=l2_regularization,
            l1_regularization=l1_regularization,
            summary_labels=summary_labels)
        self.nonlinearity = Nonlinearity(
            name=activation, summary_labels=summary_labels)
        self.nonlinearity2 = Nonlinearity(
            name=activation, summary_labels=summary_labels)
        super(EIIE, self).__init__(
            scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x0, update):
        # where window_size=50, actions=4 (giving the 3), data cols=5
        # x0 = (None,3,50,5)
        # x = (None,3,49,5)
        # x = (None,3,1,1)
        # conv1 => (None,3, 47,3)
        # conv2 => (None,3, 1, 20)
        # concat=> (None,3, 1, 21)
        # conv3 => (None,3, 1, 1)
        # concat=> (None,2, 1, 1)

        w0 = x0[:, :, :1, :1]
        x = x0[:, :, 1:, :]

        x = self.conv1.apply(x, update=update)
        x = self.nonlinearity.apply(x=x, update=update)

        x = self.conv2.apply(x, update=update)
        x = self.nonlinearity2.apply(x=x, update=update)

        x = tf.concat([x, w0], 3)
        x = self.conv3.apply(x, update=update)

        # concat on cash_bias
        cash_bias_int = 0
        # FIXME not sure how to make shape with a flexible size in tensorflow but this works for now
        # cash_bias = tf.ones(shape=(batch_size,1,1,1)) * cash_bias_int
        # cash_bias = x[:,:1,:1,:1]*0
        # x = tf.concat([cash_bias, x], 1)

        if 'activations' in self.summary_labels:
            summary = tf.summary.histogram(name='activations', values=x)
            self.summaries.append(summary)

        return x

    def tf_regularization_loss(self):
        if super(EIIE, self).tf_regularization_loss() is None:
            losses = list()
        else:
            losses = [super(EIIE, self).tf_regularization_loss()]

        if self.conv1.regularization_loss() is not None:
            losses.append(self.conv1.regularization_loss())
        if self.conv2.regularization_loss() is not None:
            losses.append(self.conv2.regularization_loss())
        if self.conv1.regularization_loss() is not None:
            losses.append(self.conv3.regularization_loss())

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None

    def get_variables(self, include_non_trainable=False):
        layer_variables = super(EIIE, self).get_variables(
            include_non_trainable=include_non_trainable)

        layer_variables += self.conv1.get_variables(
            include_non_trainable=include_non_trainable)
        layer_variables += self.conv2.get_variables(
            include_non_trainable=include_non_trainable)
        layer_variables += self.conv3.get_variables(
            include_non_trainable=include_non_trainable)

        layer_variables += self.nonlinearity.get_variables(
            include_non_trainable=include_non_trainable)
        layer_variables += self.nonlinearity.get_variables(
            include_non_trainable=include_non_trainable)

        return layer_variables


# Add our custom layer
layers['EIIE'] = EIIE

# Network as list of layers
network_spec = [
    dict(type='EIIE',
         l1_regularization=1e-8,
         l2_regularization=1e-8),
    dict(type='flatten')
]


class EIIEBaseline(NetworkBaseline):
    """
    CNN baseline (single-state) consisting of convolutional layers followed by dense layers.
    """

    def __init__(self, layers_spec, scope='eiie-baseline', summary_labels=()):
        """
        CNN baseline.
        Args:
            conv_sizes: List of convolutional layer sizes
            dense_sizes: List of dense layer sizes
        """

        super(EIIEBaseline, self).__init__(layers_spec, scope, summary_labels)


# Add our custom baseline
baselines['EIIE'] = EIIEBaseline


def main():
    data = DataMatrices(start=start, end=end,
                             start_test=start_test, end_test=end_test,
                             validation_portion=validation_portion,
                             window_size=window_size,
                             feature_list=feature_list,
                             ticker_list=ticker_list)
    fake_data = [[[0]]]

    env = PortfolioEnv(
        start_date=start,
        end_date=end,
        features_list=feature_list,
        tickers_list=ticker_list,
        scale=True,
        trading_cost=0.0025,
        window_length=window_size,
        output_mode='EIIE',
    )

    env.seed(0)
    action = np.array([0, 0.5, 0, 0.3, 0, 0.2])

    for i in range(0, 29):
        env.step(action=action)

    # wrap it in a few wrappers
    env = ConcatStates(env)
    env = SoftmaxActions(env)
    environment = TFOpenAIGymCust('CryptoPortfolioEIIE-v0', env)

    env.seed(0)

    explorations_spec = dict(
        type="epsilon_anneal",
        initial_epsilon=1.0,
        final_epsilon=0.005,
        timesteps=int(1e5),
        start_timestep=0,
    )

    # I want to use a gaussian dist instead of beta, we will apply post processing to scale everything
    # actions_spec = environment.actions.copy()
    # del actions_spec["min_value"]
    # del actions_spec["max_value"]
    # distributions_spec=dict(action=dict(type='gaussian', mean=0.25, log_stddev=np.log(5e-2)))

    # Or just use beta:
    actions_spec = environment.actions.copy()
    distributions_spec = None

    ts = datetime.utcnow().strftime('%Y%m%d_%H-%M')
    save_path = './outputs/tensorforce_PPO_crypto-%s' % ts
    log_dir = os.path.join('logs', os.path.splitext(os.path.basename(save_path))[0], 'run-' + ts)

    # https://github.com/reinforceio/tensorforce/blob/d823809df746c61471e2cba5832ab051581baf7e/docs/summary_spec.md
    summary_spec = dict(directory=log_dir,
                        steps=50,
                        labels=[
                            'configuration',
                            'gradients_scalar',
                            'regularization',
                            'inputs',
                            'losses',
                            #                             'variables'
                        ]
                        )


    agent = PPOAgent(
        states_spec=environment.states,
        actions_spec=actions_spec,
        network_spec=network_spec,
        batch_size=4096,
        saver_spec=dict(
            directory=save_path,
            steps=100000,
            #         basename=os.path.basename(save_path)
        ),
        # Agent
        states_preprocessing_spec=None,
        explorations_spec=explorations_spec,
        reward_preprocessing_spec=None,
        # BatchAgent
        keep_last_timestep=True,
        # PPOAgent
        step_optimizer=dict(
            type='adam',
            learning_rate=1e-3
        ),
        optimization_steps=10,
        # Model
        scope='ppo',
        discount=0.99,
        # DistributionModel
        distributions_spec=distributions_spec,
        entropy_regularization=0.01,  # 0 and 0.01 in baselines
        # PGModel
        baseline_mode='states',
        baseline=dict(
            type="EIIE",
            layers_spec=network_spec
            #         update_batch_size=512,
        ),  # string indicating the baseline value function (currently 'linear' or 'mlp').
        baseline_optimizer=dict(type='adam', learning_rate=0.003),
        gae_lambda=0.97,
        # PGLRModel
        likelihood_ratio_clipping=0.2,
        summary_spec=summary_spec,
        distributed_spec=None
    )

    # train
    runner = Runner(agent=agent, environment=environment)
    steps = 12e6
    env._plot = env._plot2 = env._plot3 = None
    episodes = int(steps / 30)
    runner.run(
        timesteps=steps,
        episode_finished=EpisodeFinishedTQDM(
            log_intv=1000,
            steps=steps,
            mean_of=1000,
            log_dir=log_dir,
            session=runner.agent.model.session,
        )
    )


    
if __name__ == "__main__":
    main()
    