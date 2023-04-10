import pandas as pd
import numpy as np
import cv2
from .Iterator import Iterator

default_data_path = 'data/CVPPPSegmData/'
default_split_path = 'data/CVPPPSegmData/split.csv'

class Loader(object):
    def __init__(self,
                 label_type='semantic',
                 split_path=default_split_path,
                 data_path=default_data_path,
                 batch_size=32,
                 shuffle=False,
                 num_workers=1):
        if label_type == 'semantic':
            self.col = 'sem_path'
        if label_type == 'instance':
            self.col = 'inst_path'
        self.split_path = split_path
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def load_images(self, data, channels=3):
        size = 64
        def load_image(path):
            img = cv2.imread(self.data_path + path)
            img = cv2.resize(img, (size, size))
            if channels == 1:
                return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).reshape((1, size, size))
            return img.transpose((2, 0, 1))
        return np.vectorize(load_image, signature='()->(n, m, k)')(data)

    def get_iter(self, split, images, labels, split_type):
        x = images[split == split_type]
        y = labels[split == split_type]
        return Iterator({'x': x, 'y': y}, self.batch_size, self.shuffle, self.num_workers).iter

    def load_split_data(self):
        """
        Loads split info
        Returns:
            train, dev, test: iterators
        """
        print("="*8, 'READING')
        split_df = pd.read_csv(self.split_path)
        print("-"*8, 'LOADING IMAGES')
        images = self.load_images(split_df['img_path'].to_numpy(), channels=3)
        labels = self.load_images(split_df[self.col].to_numpy(), channels=1)
        print("-"*8, 'SPLITING')
        split_types = split_df['split'].to_numpy()
        return (
            self.get_iter(split_types, images, labels, 'train'),
            self.get_iter(split_types, images, labels, 'dev'),
            self.get_iter(split_types, images, labels, 'test'),
        )