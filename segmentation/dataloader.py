import polars as pl
from .Iterator import Iterator

default_data_path = 'data/CVPPPSegmData/'
default_split_path = 'data/CVPPPSegmData/split.csv'

class Loader(object):
    def __init__(self,
                 label_type='semantic',
                 split_path=default_split_path,
                 data_path=default_data_path,
                 batch_size=1,
                 shuffle=False,
                 num_workers=1):
        self.columns = [pl.col('img_path')]
        if label_type == 'semantic' or label_type == 'any':
            self.columns.append(pl.col('sem_path'))
        if label_type == 'instance' or label_type == 'any':
            self.columns.append(pl.col('inst_path'))
        self.split_path = split_path
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def load_split_data(self):
        """
        Loads split info
        Returns:
            train, dev, test: iterators
        """
        split_df = pl.read_csv(self.split_path)
        train_dict = split_df.filter(pl.col('split') == 'train').select(self.columns).to_dict()
        dev_dict = split_df.filter(pl.col('split') == 'dev').select(self.columns).to_dict()
        test_dict = split_df.filter(pl.col('split') == 'test').select(self.columns).to_dict()
        return (
            Iterator(train_dict, self.data_path, self.batch_size, self.shuffle, self.num_workers).iter,
            Iterator(dev_dict, self.data_path, self.batch_size, self.shuffle, self.num_workers).iter,
            Iterator(test_dict, self.data_path, self.batch_size, self.shuffle, self.num_workers).iter,
        )