"""
The deep deterministic policy gradient model. Contains main training loop and deployment
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import traceback
from datetime import datetime, date

import gym
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model

from learn.root_network import RootNetwork
from rewave.environment.long_portfolio import PortfolioEnv
from rewave.learn.ddpg2.actor import Actor
from rewave.learn.ddpg2.critic import Critic
from rewave.learn.ddpg2.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise
from rewave.learn.ddpg2.replay_buffer import ReplayBuffer

from rewave.tools.paths import get_root_model_path

#from ..base_model import BaseModel


def variable_summaries(var, variable_scope):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.variable_scope(variable_scope):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def array_variable_summaries(array_var, variable_scope):
    """Attach an array of tensor summaries"""
    assert isinstance(array_var, list), 'must provide a list'
    with tf.variable_scope(variable_scope):
        for var in array_var:
            with tf.name_scope(var.name.replace(':','_')):
                variable_summaries(var,variable_scope)

class DDPG(object):
    def __init__(self, env, sess, actor_noise, action_dim, obs_normalizer=None, action_processor=None,
                 predictor_type="cnn", use_batch_norm=False,
                 model_save_path='weights/ddpg', summary_path='results/ddpg2/'):
        self.config = {
              "episode": 1000, #100
              "max step": 100, #1000
              "buffer size": 100000,
              "batch size": 32,
              "tau": 0.02,
              "gamma": 0.1,
              "actor learning rate": 0.001,
              "critic learning rate": 0.001,
              "seed": 1337
            }
        assert self.config['max step'] > self.config['batch size'], 'Max step must be bigger than batch size'

        self.actor_learning_rate = self.config["actor learning rate"]
        self.critic_learning_rate = self.config["critic learning rate"]
        self.tau = self.config["tau"]
        self.gamma = self.config["gamma"]

        self.action_processor = action_processor

        np.random.seed(self.config['seed'])
        if env:
            env.seed(self.config['seed'])
        self.model_save_path = model_save_path + '/checkpoint.ckpt'
        self.summary_path = summary_path + "/" + datetime.now().strftime("%Y-%m-%d-%H%M%S")
        self.root_model_save_path = model_save_path + '/root_model.h5'
        self.sess = sess
        # if env is None, then DDPG just predicts
        self.env = env

        self.actor_noise = actor_noise
        # share state input
        has_complex_state = (isinstance(self.env.observation_space, gym.spaces.Dict) or isinstance(self.env.observation_space, gym.spaces.Tuple))
        if obs_normalizer and has_complex_state:
            state_input = Input(shape=self.env.observation_space.spaces[obs_normalizer].shape, name="state_input")
        else:
            state_input = Input(shape=self.env.observation_space.shape, name="state_input")

        target_state_input = Input(shape=self.env.observation_space.spaces[obs_normalizer].shape, name="target_state_input")
        self.obs_normalizer = obs_normalizer

        # feature extraction
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        root_net = RootNetwork(inputs=state_input,
                              predictor_type=predictor_type,
                              use_batch_norm=use_batch_norm).net

        variable_summaries(root_net, "Root_Output")

        self.root_model = Model(state_input, root_net)

        array_variable_summaries(self.root_model.layers[1].weights, "Root_Input_1")
        array_variable_summaries(self.root_model.layers[2].weights, "Root_Input_2")
        array_variable_summaries(self.root_model.layers[-1].weights, "Root_Output_2")

        target_root_net = RootNetwork(inputs=target_state_input,
                              predictor_type=predictor_type,
                              use_batch_norm=use_batch_norm).net

        self.target_root_model = Model(target_state_input, target_root_net)
        # ===================================================================== #
        #                               Actor Model                             #
        # Chain rule: find the gradient of chaging the actor network params in  #
        # getting closest to the final value network predictions, i.e. de/dA    #
        # Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
        # ===================================================================== #

        self.actor_state_input, self.actor_model = Actor(state_input=state_input, root_net=root_net, action_dim=action_dim).references()
        _, self.target_actor_model = Actor(state_input=target_state_input, root_net=target_root_net, action_dim=action_dim).references()

        self.actor_critic_grad = tf.placeholder(tf.float32,
                                                [None, self.env.action_space.shape[
                                                    0]])  # where we will feed de/dC (from critic)
        # summary
        array_variable_summaries(self.actor_model.layers[-1].weights, "Actor_Output")

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output,
                                        actor_model_weights, -self.actor_critic_grad)  # dC/dA (from actor)

        tf.summary.histogram("Actor_Critic_Grad", self.actor_critic_grad)

        grads = zip(self.actor_grads, actor_model_weights)

        self.optimize = tf.train.AdamOptimizer(self.actor_learning_rate).apply_gradients(grads)

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #

        self.critic_state_input, self.critic_action_input, self.critic_model = Critic(state_input=state_input,
                                                                                      root_net=root_net,
                                                                                      action_dim=action_dim,
                                                                                      lr=self.critic_learning_rate).references()
        array_variable_summaries(self.critic_model.layers[-1].weights, "Critic_Output")

        _, _, self.target_critic_model = Critic(state_input=target_state_input,
                                            root_net=target_root_net,
                                            action_dim=action_dim,
                                            lr=self.critic_learning_rate).references()

        self.critic_grads = tf.gradients(self.critic_model.output,
                                         self.critic_action_input)  # where we calcaulte de/dC for feeding above


        # summary
        #self.summary_ops, self.summary_vars = build_summaries(action_dim=action_dim)
        with tf.variable_scope("Global"):
            self.episode_reward = tf.Variable(0.)
            tf.summary.scalar("Reward", self.episode_reward)
            self.episode_min_reward = tf.Variable(0.)
            tf.summary.scalar("Min_Reward", self.episode_min_reward)
            self.episode_ave_max_q = tf.Variable(0.)
            tf.summary.scalar("Qmax_Value", self.episode_ave_max_q)
            self.critic_loss = tf.Variable(0.)
            tf.summary.scalar("Critic_loss", self.critic_loss)
            self.ep_base_action = tf.Variable(initial_value=self.env.sim.w0)
            tf.summary.histogram("Action_base", self.ep_base_action)
            self.ep_action = tf.Variable(initial_value=self.env.sim.w0)
            tf.summary.histogram("Action", self.ep_action)

        self.merged = tf.summary.merge_all()

        # Initialize for later gradient calculations
        self.sess.run(tf.global_variables_initializer())


        # ========================================================================= #
        #                         Target Model Updating                             #
        # ========================================================================= #

    def _update_actor_target(self):
        weights = self.actor_model.get_weights()
        target_weights = self.target_actor_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_actor_model.set_weights(target_weights)

    def _update_critic_target(self):
        weights = self.critic_model.get_weights()
        target_weights = self.target_critic_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_critic_model.set_weights(target_weights)

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()


    # ========================================================================= #
    #                              Model Predictions                            #
    # ========================================================================= #

    def act(self, cur_state):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.actor_model.predict(cur_state).reshape((self.env.action_space.shape[0],))

    def initialize(self, load_weights=True, verbose=True):
        """ Load training history from path. To be add feature to just load weights, not training states

        """
        if load_weights:
            try:
                variables = tf.global_variables()
                param_dict = {}
                saver = tf.train.Saver()
                saver.restore(self.sess, self.model_save_path)
                for var in variables:
                    var_name = var.name[:-2]
                    if verbose:
                        print('Loading {} from checkpoint. Name: {}'.format(var.name, var_name))
                    param_dict[var_name] = var
            except:
                traceback.print_exc()
                print('Build model from scratch')
                self.sess.run(tf.global_variables_initializer())
        else:
            print('Build model from scratch')
            self.sess.run(tf.global_variables_initializer())


    def _train_actor(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, _ = sample
            predicted_action = self.actor_model.predict(cur_state)
            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input: cur_state,
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: cur_state,
                self.actor_critic_grad: grads
            })

    def _train_critic(self, samples):
        batch_size = self.config['batch size']
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if not done:
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict(
                    [new_state, target_action])[0][0]
                reward += self.gamma * future_reward
            history_critic = self.critic_model.fit([cur_state, action], reward, verbose=0, batch_size=batch_size)
            # print("reward = ", reward, "/", self.critic_model.predict([cur_state, action]))
            # for layer in self.critic_model.layers:
            # print(layer, " weights = ", layer.get_weights())
            return history_critic

    def train(self, save_every_episode=1, verbose=True, debug=False):
        writer = tf.summary.FileWriter(self.summary_path, self.sess.graph)
        np.random.seed(self.config['seed'])
        num_episode = self.config['episode']
        batch_size = self.config['batch size']
        gamma = self.config['gamma']
        self.buffer = ReplayBuffer(self.config['buffer size'])

        # main training loop
        for i in range(num_episode):
            if verbose and debug:
                print("Episode: {} Replay Buffer  {}".format(i, self.buffer.count))

            previous_observation = self.env.reset()
            if self.obs_normalizer:
                previous_observation = previous_observation[self.obs_normalizer]

            ep_reward = 0.0
            episode_min_reward = 0.0
            ep_ave_max_q = 0.0
            # keeps sampling until done
            for j in range(self.config['max step']):
                base_action = self.actor_model.predict(np.expand_dims(previous_observation, axis=0)).squeeze(
                    axis=0)

                action = base_action + self.actor_noise()

                # normalize action
                action = np.clip(action, 0.0, 1.0)
                action /= action.sum()

                action_take = action
                if self.action_processor:
                    action_take = self.action_processor(action)

                # step forward
                observation, reward, done, _ = self.env.step(action_take)

                if self.obs_normalizer:
                    observation = observation[self.obs_normalizer]

                # add to buffer
                self.buffer.add(previous_observation, action, reward, done, observation)

                # keep track of loss
                critic_loss = 0.0

                if self.buffer.size() >= batch_size:
                    # ========================================================================= #
                    #                         batch update                                      #
                    # ========================================================================= #
                    s_batch, a_batch, r_batch, t_batch, s2_batch = self.buffer.sample_batch(batch_size)
                    # Calculate targets
                    target_q = self.target_critic_model.predict([s2_batch, self.target_actor_model.predict(s2_batch)])

                    y_i = []
                    for k in range(batch_size):
                        if t_batch[k]:
                            y_i.append(r_batch[k])
                        else:
                            y_i.append(r_batch[k] + gamma * target_q[k][0])

                    # Update the critic given the targets
                    critic_loss += self.critic_model.train_on_batch([s_batch, a_batch], np.reshape(y_i, (batch_size, 1)))[0]

                    #target_history = self.target_critic_model.fit([s_batch, a_batch], np.reshape(y_i, (batch_size, 1)))

                    predicted_q_value =  self.critic_model.predict([s_batch, a_batch])
                    # predicted_target_q_value = self.target_critic_model.predict([s_batch, a_batch])

                    a_for_grad = self.actor_model.predict(s_batch)

                    ep_ave_max_q += np.amax(predicted_q_value)

                    # Update the actor policy using the sampled gradient
                    a_outs = self.actor_model.predict(s_batch)
                    grads = self.sess.run(self.critic_grads, feed_dict={
                        self.critic_state_input: s_batch,
                        self.critic_action_input: a_outs
                    })[0]

                    self.sess.run(self.optimize, feed_dict={
                        self.actor_state_input: s_batch,
                        self.actor_critic_grad: grads
                    })

                    # Update target networks
                    self.update_target()

                ep_reward += reward
                episode_min_reward = min(reward, episode_min_reward)
                previous_observation = observation

                if done or j == self.config['max step'] - 1:

                    # do summary preparation
                    merged, _ = self.sess.run([self.merged, self.optimize], feed_dict={
                        self.actor_state_input: s_batch,
                        self.actor_critic_grad: grads,
                        self.episode_reward: ep_reward / (float(j) if j != 0 else 1.0),
                        self.critic_loss: critic_loss,
                        self.episode_min_reward: episode_min_reward,
                        self.ep_action: action,
                        self.ep_base_action: base_action,
                        self.episode_ave_max_q: ep_ave_max_q / (float(j) if j != 0 else 1.0)

                    })

                    writer.add_summary(merged, i)
                    writer.flush()

                    print('Episode: {:d}, Average Reward: {:.2f}, Average Qmax: {:.4f}'.format(i, (ep_reward / float(j)), (ep_ave_max_q / float(j))))
                    print('---top indice {}, top 3 base actions {}'.format(np.where(base_action == base_action.max())[0][0], sorted(base_action)[-3:]))
                    #print('Action: norm {}, values {}'.format(action.sum(), action))
                    #print('---Base Action: norm {}, values {}'.format(base_action.sum(), base_action))
                    break

        self.save_model(verbose=True)
        print('Finish.')


    def predict(self, observation):
        """ predict the next action using actor model, only used in deploy.
            Can be used in multiple environments.

        Args:
            observation: (batch_size, num_stocks + 1, window_length)

        Returns: action array with shape (batch_size, num_stocks + 1)

        """

        if self.obs_normalizer:
            observation = self.obs_normalizer(observation)
        action = self.actor.predict(observation)
        if self.action_processor:
            action = self.action_processor(action)
        return action

    def predict_single(self, observation):
        """ Predict the action of a single observation

        Args:
            observation: (num_stocks + 1, window_length)

        Returns: a single action array with shape (num_stocks + 1,)

        """
        if self.obs_normalizer and isinstance(observation, dict):
            observation = observation[self.obs_normalizer]
        action = self.actor_model.predict(np.expand_dims(observation, axis=0)).squeeze(axis=0)
        if self.action_processor:
            action = self.action_processor(action)
        return action

    def save_model(self, verbose=False):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path, exist_ok=True)

        saver = tf.train.Saver()
        model_path = saver.save(self.sess, self.model_save_path)
        print("Model saved in %s" % model_path)


#tickers_list = ['AAPL', 'ATVI', 'CMCSA', 'COST', 'CSX', 'DISH', 'EA', 'EBAY', 'FB', 'GOOGL', 'HAS', 'ILMN', 'INTC', 'MAR', 'REGN', 'SBUX']

tickers_list = ['NLSN', 'LEG', 'JPM', 'NFX', 'IT', 'DISH', 'EA', 'EBAY', 'FB', 'OXY', 'HAS', 'ILMN', 'DWDP', 'AAPL', 'UPS', 'VRTX']

window_length = 15
predictor_type="cnn"
use_batch_norm = False

def test_main():
    print("STARTING MAIN")
    start_date = date(2008, 1, 1)
    end_date = date(2017, 1, 1)
    features_list = ['open', 'high', 'low', 'close']

    # setup environment
    num_training_time = 100
    env = PortfolioEnv(start_date, end_date,
                       window_length,
                       tickers_list, features_list, batch_size=num_training_time)

    #variable_scope ="ddpg"
    nb_assets = len(tickers_list)+1
    action_dim = env.action_space.shape[0]
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.ones(action_dim)/nb_assets)

    #with tf.variable_scope(variable_scope):
    tf.reset_default_graph()
    sess = tf.Session()

    with sess.as_default():

        # Missing this was the source of one of the most challenging an insidious bugs that I've ever encountered.
        # it was impossible to save the model.
        K.set_session(sess)

        ddpg_model = DDPG(env, sess, actor_noise, action_dim=action_dim, obs_normalizer="history", predictor_type=predictor_type, use_batch_norm=use_batch_norm)
        ddpg_model.initialize(load_weights=False)
        ddpg_model.train()

def test_predict():
    print("STARTING PREDICT")
    start_date = date(2017, 1, 1)
    end_date = date(2018, 1, 1)
    features_list = ['open', 'high', 'low', 'close']

    # setup environment
    num_training_time = 200
    env = PortfolioEnv(start_date, end_date,
                       window_length,
                       tickers_list, features_list,
                       trading_cost=0.0,
                       batch_size=num_training_time, buffer_bias_ratio=0.0)

    #variable_scope ="ddpg"
    action_dim = env.action_space.shape[0]
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

    #with tf.variable_scope(variable_scope):
    tf.reset_default_graph()
    sess = tf.Session()

    with sess.as_default():
        # Missing this was the source of one of the most challenging an insidious bugs that I've ever encountered.
        # it was impossible to save the model.
        K.set_session(sess)

        ddpg_model = DDPG(env, sess, actor_noise, action_dim=action_dim, obs_normalizer='history', predictor_type=predictor_type, use_batch_norm=use_batch_norm)
        ddpg_model.initialize(load_weights=True)


        observation = env.reset()
        done= False
        while not done:
            action = ddpg_model.predict_single(observation)
            observation, reward, done, infos = env.step(action)
            print(infos['date'],infos['portfolio_value'], infos['market_value'], infos['weight_cash'], infos['weight_AAPL'])
        df = env.df_info()
        print(df.iloc[-1,:])
        print(df.describe())
        env.render(mode='notebook')

if __name__ == '__main__':
    #test_main()
    test_predict()