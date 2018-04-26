from __future__ import absolute_import
from unittest import TestCase
import unittest
from rewave.marketdata.replaybuffer import ReplayBuffer, Experience
from datetime import date
import logging

class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.start_index=0
        self.end_index=250
        self.batch_size=50
        self.is_permed=False
        self.sample_bias=1e-5
        self.replayBuffer = ReplayBuffer(start_index=self.start_index, end_index=self.end_index,
                                         batch_size=self.batch_size, is_permed=self.is_permed,
                                         sample_bias=self.sample_bias)
        self.replayBuffer_Permuted = ReplayBuffer(start_index=self.start_index, end_index=self.end_index,
                                         batch_size=self.batch_size, is_permed=True,
                                         sample_bias=self.sample_bias)

        self.replayBuffer_Full = ReplayBuffer(start_index=self.start_index, end_index=self.end_index,
                                                  batch_size=0, is_permed=False,
                                                  sample_bias=self.sample_bias)


    def test_batch_size(self):
        batch = self.replayBuffer.next_experience_batch()
        self.assertEquals(len(batch), self.batch_size,"batch size has to be as long as the parameter batch_size")

    def test_random(self):
        batch_1 = self.replayBuffer.next_experience_batch()
        batch_2 = self.replayBuffer.next_experience_batch()
        self.assertNotEqual(batch_1[0].state_index, batch_2[0].state_index, "random batch is not working")

    def test_permutation(self):
        batch = self.replayBuffer.next_experience_batch()
        self.assertEqual(batch[1].state_index - batch[0].state_index, 1, "permutation in batch")

        batch = self.replayBuffer_Permuted.next_experience_batch()
        self.assertNotEqual(batch[1].state_index-batch[0].state_index, 1, "permutation for batch is not working")

    def test_append(self):
        state_index = self.end_index +1
        self.replayBuffer.append_experience(state_index)
        self.assertEquals(self.end_index-self.start_index +2, len(self.replayBuffer._experiences) )

    def test_bias_zero(self):
        replayBuffer = ReplayBuffer(start_index=self.start_index, end_index=self.end_index,
                                              batch_size=0, is_permed=False,
                                              sample_bias=self.sample_bias)
        self.assertEqual(0, replayBuffer._sample_bias, "bias is null for full replay")

    def test_full_batchsize(self):
        total = self.end_index - self.start_index +1
        self.assertEqual(total, self.replayBuffer_Full._batch_size, "all experiences if batch size equal zero")
        self.assertEqual(0, self.replayBuffer_Full._sample_bias, "bias null if full batchsize")
        self.assertEqual(0, self.replayBuffer_Full._sample(self.start_index, self.end_index, 0), "must start at zero in full replay")
