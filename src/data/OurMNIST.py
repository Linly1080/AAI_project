import os
import random
from collections import defaultdict

import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torch.distributions.categorical import Categorical


class OurMNIST(Dataset):
    def __init__(self, file_path, data_config, target=False):

        self.label_to_y = self.load_data_config(data_config)

        self.data = {
            'train': self.load_data(file_path=file_path, train=True,
                                    label_to_y=self.label_to_y),
            'test': self.load_data(file_path=file_path, train=False,
                                   label_to_y=self.label_to_y),
        }

        # define the training, val and testing environments
        self.envs = []

        # use the first half for train env 0
        self.envs.append(self.create_env(
            data=self.data['train'], bias_ratio=0.1, start_ratio=0, end_ratio=0.4))

        # use the second half for train env 1
        self.envs.append(self.create_env(
            data=self.data['train'], bias_ratio=0.2 if target else 0.1,
            start_ratio=0.4, end_ratio=0.8))

        # use the first half for val env
        self.envs.append(self.create_env(
            data=self.data['train'], bias_ratio=0.1, start_ratio=0.8, end_ratio=1))

        # use the second half for test env
        self.envs.append(self.create_env(
            data=self.data['test'], bias_ratio=0.9, start_ratio=0, end_ratio=1))

        self.length = sum([len(env['images']) for env in self.envs])

        # not evaluating worst-case performance of mnist
        self.val_att_idx_dict = None
        self.test_att_idx_dict = None


    def load_data_config(self, data_config):
        '''
            The first segment represent the digits
            The second segment represent the colors that are corerlated with each
            of the digits
            The last segment represent the maximum number of colors

            Examples:
            EVEN: MNIST_02468_01234_5
            ODD: MNIST_13579_01234_5
        '''
        label, color, max_c = tuple(data_config.split('_'))
        y_list = [int(y) for y in label]

        if y_list is not None:
            # We have specified the digits that we want to classify in y_list
            # Here we map each digit into a 0-based index list
            label_to_y = {}
            for i, y in enumerate(y_list):
                label_to_y[y] = i

        else:
            # use all ten digits if not specified
            label_to_y = dict(zip(list(range(10)), list(range(10))))

        return label_to_y


    def load_data(self, file_path, train, label_to_y):
        # load the data based on the current label_to_dict
        # mnist = datasets.MNIST(file_path, train=train, download=True)

        data = defaultdict(list)

        path = os.path.join(file_path, 'train' if train else 'val')
        for y in label_to_y.keys():
            file_dir = os.path.join(path,str(y))
            for file_name in os.listdir(file_dir):
                file_path = os.path.join(file_dir, file_name)
                x = np.load(file_path)
                x = torch.from_numpy(x)
                data[label_to_y[int(y)]].append(x)

        # shuffle data
        random.seed(0)
        for k, v in data.items():
            random.shuffle(v)
            data[k] = torch.stack(v, dim=0)

        return data


    def create_env(self, data, bias_ratio, start_ratio, end_ratio):
        '''
            Create an environment using data from the start_ratio to the end_ratio
        '''
        images = []
        labels = []
        cor = []

        for cur_label, cur_images in data.items():
            start = int(start_ratio * len(cur_images))
            end = int(end_ratio * len(cur_images))

            mean1 = torch.sum(cur_images[start:end], dim=3)
            mean2 = torch.sum(mean1, dim=2)
            max_values, _ = torch.max(mean2, dim=1)
            channel_indices = mean2.argmax(dim=1)
            #
            # cur_images = torch.sum(cur_images, dim=1)
            # cur_images = torch.unsqueeze(cur_images, 1)

            images.append(cur_images[start:end])
            labels.append((torch.ones(end-start) * cur_label).long())
            cor.append(channel_indices)

        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)
        cor = torch.cat(cor, dim=0)

        print(images.shape)
        print(labels.shape)
        print(cor.shape)

        idx_dict = defaultdict(list)
        for i in range(len(images)):
            idx_dict[int(labels[i])].append(i)

        idx_list = list(range(len(images)))
        return {
            'images': images,
            'labels': labels,
            'idx_dict': idx_dict,
            'idx_list': idx_list,
            'cor': cor,
        }


    def __len__(self):
        return self.length

    def __getitem__(self, keys):
        '''
            @params [support, query]
            support=[(label, y, idx, env)]
            query=[(label, y, idx, env)]
        '''

        # without reindexing y
        idx = []
        for key in keys:
            env_id = int(key[1])
            idx.append(key[0])

        return {
            'X': self.envs[env_id]['images'][idx],
            'Y': self.envs[env_id]['labels'][idx],
            'C': self.envs[env_id]['cor'][idx],
            'idx': torch.tensor(idx).long(),
        }


    def get_all_y(self, env_id):
        return self.envs[env_id]['labels'].tolist()


    # def get_all_c(self, env_id):
    #     return self.envs[env_id]['cor'].tolist()
