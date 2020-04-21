from __future__ import print_function
import shutil
import gzip
import os
import pickle
import random
from collections import namedtuple
import torch
import numpy as np

Transitions = namedtuple('Transitions',
                         ('image', 'numerical_state', 'action', 'reward', 'next_image', \
                         'next_numerical_state', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity, batch_size): #capacity of the one file
        self.capacity = capacity
        self.memory = []
        self.batch_size = batch_size
        self.position = 0
        self.buffer_num = 1
        self.buffer_available = 1
        self.pointer = 0
        self.low = 0
        self.flag = False
        self.indices = np.arange(self.capacity)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.new = False

    global COUNTER
    COUNTER = 0
    def push(self, *args):
        self.new = False
        global COUNTER
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transitions(*args)
        self.position = (self.position + 1) % self.capacity
        self.pointer += 1
        if self.pointer > self.capacity:
            self.pointer = 0
            self.flag = True
            self.buffer_num += 1

        if self.flag:
            COUNTER += 1

        if COUNTER > self.batch_size:
            self.buffer_available += 1
            self.flag = False
            COUNTER = 0
            self.new = True

        if self.buffer_num - 1500 > 0:
            self.low = self.buffer_num - 1500
            try:
                os.remove(r"/home/george/Desktop/Github/Reinforcement_Learning/" \
                    r"Datasets/my_dataset"+str(self.buffer_num - 1500)+'.pkl.gz')
            except:
                print('Cannot erase the iteration from the buffer')

    def sample(self):
        buf = np.random.randint(self.low, self.buffer_available, 1)[0]
        if buf == self.buffer_available-1 and not self.flag: #if the last
            ind = random.sample(self.indices[:self.pointer], \
                self.batch_size) #the pointer has become 0
        else:
            ind = random.sample(self.indices, self.batch_size)
        objects = []
        j = 0
        if buf != self.buffer_available-1:
            gzip_file = gzip.open(r"/home/george/Desktop/Github/Reinforcement_Learning/" \
                r"Datasets/my_dataset"+str(buf+1)+'.pkl.gz', 'rb')
            data_file = open(r"/home/george/Desktop/Github/Reinforcement_Learning/" \
                r"Datasets/my_dataset"+str(buf+1)+'.pkl', 'wb')
            shutil.copyfileobj(gzip_file, data_file)
            data_file.close()
            gzip_file.close()

        with (open(r"/home/george/Desktop/Github/Reinforcement_Learning/" \
            r"Datasets/my_dataset"+str(buf+1)+'.pkl', 'rb')) as openfile:
            while j < self.capacity:
                try:
                    file = pickle.load(openfile)
                    if j in ind:
                        objects.append(file)
                    else:
                        del file
                    if len(objects) == self.batch_size:
                        break
                except EOFError:
                    break
                j += 1
        if buf != self.buffer_available-1:
            os.remove(r"/home/george/Desktop/Github/Reinforcement_Learning/" \
                r"Datasets/my_dataset"+str(buf+1)+'.pkl')
        image = []
        numerical_state = []
        action = []
        reward = []
        next_image = []
        next_numerical_state = []
        done = []
        for obj in objects:
            image.append(obj.image)
            numerical_state.append(obj.numerical_state)
            action.append(obj.action)
            reward.append(obj.reward)
            next_image.append(obj.next_image)
            next_numerical_state.append(obj.next_numerical_state)
            done.append(obj.done)

        image_state_batch = torch.from_numpy(np.asarray(image)). \
            squeeze(1).permute(0, 3, 1, 2).float().to(self.device)
        numerical_state_batch = torch.FloatTensor(np.asarray \
            (numerical_state)).squeeze(1).to(self.device)
        action_batch = torch.FloatTensor(np.asarray(action)).squeeze(1).to(self.device)
        reward_batch = torch.FloatTensor(np.asarray(reward)).squeeze().to(self.device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                next_image)), device=self.device, dtype=torch.uint8)
        non_final_next_image = torch.from_numpy(np.asarray([img for img in next_image \
            if img is not None])).squeeze(1).permute(0, 3, 1, 2).float().to(self.device)
        non_final_next_numerical_state = torch.FloatTensor(np.asarray([state for \
            state in next_numerical_state if state is not None])).squeeze(1).to(self.device)
        done_batch = torch.FloatTensor(np.float32(done)).squeeze().to(self.device)

        return image_state_batch, numerical_state_batch, action_batch, reward_batch, \
            non_final_mask, non_final_next_image, non_final_next_numerical_state, done_batch

    def len(self):
        return len(self.memory)

    def empty(self):
        self.memory = []
        self.position = 0

    def store_at_disk(self):
        if self.new:
            data_file = open(r"/home/george/Desktop/Github/Reinforcement_Learning/" \
                r"Datasets/my_dataset"+str(self.buffer_num-1)+'.pkl', 'rb')
            data = data_file.read()
            data_gzip_file = gzip.open(r"/home/george/Desktop/Github/Reinforcement_Learning/" \
                r"Datasets/my_dataset" + str(self.buffer_num-1)+'.pkl.gz', 'wb')
            data_gzip_file.write(data)
            data_gzip_file.close()
            data_file.close()
            os.remove(r"/home/george/Desktop/Github/Reinforcement_Learning/" \
                r"Datasets/my_dataset"+str(self.buffer_num-1)+'.pkl')
        else:
            data_file = open(r"/home/george/Desktop/Github/Reinforcement_Learning/" \
                r"Datasets/my_dataset" + str(self.buffer_num)+'.pkl', 'ab')
            pickle.dump(self.memory[0], data_file, -1)
            data_file.close()
