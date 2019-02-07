import random
from collections import deque


class ReplayBuffer(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.cur_size = 0
        self.buffer = deque()

    def __len__(self):
        return self.cur_size

    def add(self, experience):
        if self.cur_size < self.max_size:
            self.buffer.append(experience)
            self.cur_size += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def get_batch(self, size):
        sample_size = size if size <= self.cur_size else self.cur_size
        return random.sample(self.buffer, sample_size)

    def clear(self):
        self.buffer.clear()
        self.cur_size = 0
