from __future__ import division,absolute_import,print_function
import numpy as np
import logging


class ReplayBuffer:
    """
    Extract a list of indices for the batch. Can be extended for online usage with additional indice
    """
    def __init__(self, start_index, end_index, batch_size, is_permed, sample_bias=1e-5):
        """
        :param start_index: start index of the training set on the global data matrices
        :param end_index: end index of the training set on the global data matrices
        """
        self._experiences = [Experience(i) for i in range(start_index, end_index)]
        self._is_permed = is_permed
        # NOTE: in order to achieve the previous w feature
        self._batch_size = batch_size
        self._sample_bias = sample_bias
        logging.debug("buffer_bias is %f" % sample_bias)

    def append_experience(self, state_index):
        self._experiences.append(Experience(state_index))
        logging.debug("a new experience, indexed by %d, was appended" % state_index)

    def _sample(self, start, end, bias):
        """
        @:param end: is excluded
        @:param bias: value in (0, 1)
        """
        # TODO: deal with the case when bias is 0
        # make sure we have a random number with the seed value
        np.random.seed(None)
        ran = np.random.geometric(bias)
        while ran > end - start:
            ran = np.random.geometric(bias)
        result = end - ran
        return result

    def next_experience_batch(self):
        # First get a start point randomly
        batch = []
        if self._is_permed:
            for i in range(self._batch_size):
                batch.append(self._experiences[self._sample(self._experiences[0].state_index,
                                                              self._experiences[-1].state_index,
                                                              self._sample_bias)])
        else:
            batch_start = self._sample(0, len(self._experiences) - self._batch_size,
                                        self._sample_bias)
            batch = self._experiences[batch_start:batch_start+self._batch_size]
        return batch


class Experience:
    def __init__(self, state_index):
        self.state_index = int(state_index)
