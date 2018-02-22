from __future__ import absolute_import
from unittest import TestCase
import unittest
from rewave.environment.openai_gym_cust import OpenAIGymCustom
from datetime import date
import logging


import gym
from gym.spaces import Discrete, Dict, Tuple
from gym.spaces import Box

import numpy as np
from tensorforce import TensorForceError
from tensorforce.environments import Environment


import numpy as np

class TestOpenAIGymCustom(unittest.TestCase):

    def test_states_discrete(self):
        gym_spaces = Discrete(10)
        tensorforce_spaces = {'shape': (), 'type': 'int'}
        env = OpenAIGymCustom(gym_id="CartPole-v0")
        self.assertEqual(env.state_from_space(gym_spaces), tensorforce_spaces, "conversion of states is not correct")

    def test_states_box(self):
        gym_spaces = Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
        tensorforce_spaces = {'shape': (1,), 'type': 'float'}
        env = OpenAIGymCustom(gym_id="CartPole-v0")
        self.assertEqual(env.state_from_space(gym_spaces), tensorforce_spaces, "conversion of states is not correct")

    def test_states_tuple(self):
        gym_spaces = Tuple({'history': Box(low=0.0, high=1.0, shape=(6, 50, 3), dtype=np.float32),
                       'weights': Box(low=1.0, high=1.0, shape=(6,), dtype=np.float32),
                       'weights2': Box(low=1.0, high=1.0, shape=(6,), dtype=np.float32)})
        tensorforce_spaces = {'history': {'shape': (6, 50, 3), 'type': 'float'},
                             'weights': {'shape': (6,), 'type': 'float'},
                             'weights2': {'shape': (6,), 'type': 'float'}}
        env = OpenAIGymCustom(gym_id="CartPole-v0")
        self.assertEqual(env.state_from_space(gym_spaces), tensorforce_spaces, "conversion of states is not correct")

    def test_states_dict(self):
        gym_spaces = Dict({'history': Box(low=0.0, high=1.0, shape=(6, 50, 3), dtype=np.float32),
                       'weights': Box(low=1.0, high=1.0, shape=(6,), dtype=np.float32),
                       'weights2': Box(low=1.0, high=1.0, shape=(6,), dtype=np.float32)})
        tensorforce_spaces = {'history': {'shape': (6, 50, 3), 'type': 'float'},
                             'weights': {'shape': (6,), 'type': 'float'},
                             'weights2': {'shape': (6,), 'type': 'float'}}
        env = OpenAIGymCustom(gym_id="CartPole-v0")
        self.assertEqual(env.state_from_space(gym_spaces), tensorforce_spaces, "conversion of states is not correct")

    ############
    # Actions
    ############

    def test_actions_discrete(self):
        gym_spaces = Discrete(10)
        tensorforce_spaces = {'type': 'int', 'num_actions': 10}
        env = OpenAIGymCustom(gym_id="CartPole-v0")
        self.assertEqual(env.action_from_space(gym_spaces), tensorforce_spaces, "conversion of actions is not correct")

    def test_actions_box(self):
        gym_spaces = Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
        tensorforce_spaces = {'type': 'float', 'shape': (1,), 'min_value': -10.0, 'max_value': 10.0}
        env = OpenAIGymCustom(gym_id="CartPole-v0")
        self.assertEqual(env.action_from_space(gym_spaces), tensorforce_spaces, "conversion of actions is not correct")

    def test_actions_tuple(self):
        gym_spaces = Tuple({'history': Box(low=0.0, high=1.0, shape=(6, 50, 3), dtype=np.float32),
                       'weights': Box(low=1.0, high=1.0, shape=(6,), dtype=np.float32),
                       'weights2': Box(low=1.0, high=1.0, shape=(6,), dtype=np.float32)})
        tensorforce_spaces = {'history': {'shape': (6, 50, 3), 'type': 'float'},
                             'weights': {'shape': (6,), 'type': 'float'},
                             'weights2': {'shape': (6,), 'type': 'float'}}
        env = OpenAIGymCustom(gym_id="CartPole-v0")
        env.action_from_space(gym_spaces)

    def test_actions_dict(self):
        gym_spaces = Dict({'history': Box(low=0.0, high=1.0, shape=(6, 50, 3), dtype=np.float32),
                       'weights': Box(low=1.0, high=1.0, shape=(6,), dtype=np.float32),
                       'weights2': Box(low=1.0, high=1.0, shape=(6,), dtype=np.float32)})
        tensorforce_spaces = {'history': {'shape': (6, 50, 3), 'type': 'float'},
                             'weights': {'shape': (6,), 'type': 'float'},
                             'weights2': {'shape': (6,), 'type': 'float'}}
        env = OpenAIGymCustom(gym_id="CartPole-v0")
        env.action_from_space(gym_spaces)
